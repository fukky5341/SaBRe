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
execution time: IAR + LP analysis = 13.88 + 33.24 = 47.12 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.88 seconds, max iter: 100)

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
Binary search time: 146.63 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_random_Z) starts
Time budget: 3406.25 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3607291, upper bound: 1.3704543
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3704541, upper bound: 1.3607290
time: 4.21 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.32 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.32
Output dim: 7, lower bound: -1.3607291, upper bound: 1.3704543
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.32
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

Time for backsubstitution: 13.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 411

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 604

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3574775, upper bound: 1.3575030
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3477725, upper bound: 1.3672083
time: 3.65 seconds

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

Time for backsubstitution: 13.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3605092, upper bound: 1.3504287
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3601236, upper bound: 1.3508146
time: 4.13 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 21.33 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.33
Output dim: 7, lower bound: -1.3574775, upper bound: 1.3575030
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.33
Output dim: 7, lower bound: -1.3477725, upper bound: 1.3672083
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 21.33
Output dim: 7, lower bound: -1.3605092, upper bound: 1.3504287
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 21.33
Output dim: 7, lower bound: -1.3601236, upper bound: 1.3508146

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3036838, 2.3079376
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6245008, 2.6161971
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1921668, 2.1893835
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8593640, 1.8601904
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8796768, 1.8807960
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4360585, 2.4321928
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1936131, 2.2018666
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6066194, 2.5923567
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 13.12 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1760

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3364326, upper bound: 1.3360599
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3364326, upper bound: 1.3360599
time: 3.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.2958026, 2.3158193
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6241231, 2.6165745
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1905656, 2.1909847
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8656154, 1.8539391
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8764315, 1.8840411
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4335041, 2.4347472
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1933861, 2.2020936
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6043305, 2.5946441
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 13.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 921

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 634

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3477406, upper bound: 1.3658098
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3463981, upper bound: 1.3671677
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1

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

Time for backsubstitution: 13.08 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1734

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3436648, upper bound: 1.3359103
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3457307, upper bound: 1.3338361
time: 4.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2

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

Time for backsubstitution: 13.11 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 3125

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3432147, upper bound: 1.3363455
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3452976, upper bound: 1.3342848
time: 4.45 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 21.95 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.3364326, upper bound: 1.3360599
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.3364326, upper bound: 1.3360599
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.3477406, upper bound: 1.3658098
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.3463981, upper bound: 1.3671677
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.3436648, upper bound: 1.3359103
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.3457307, upper bound: 1.3338361
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.3432147, upper bound: 1.3363455
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 21.95
Output dim: 7, lower bound: -1.3452976, upper bound: 1.3342848

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3130426, 2.3219528
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6284180, 2.6174965
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1943645, 2.1930246
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8722839, 1.8577795
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905039, 1.8941026
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4404769, 2.4363184
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1946554, 2.2031598
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6131673, 2.6028748
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 13.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1949

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1096

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3195125, upper bound: 1.3240874
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3243982, upper bound: 1.3192040
time: 3.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3098173, 2.3221960
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6254225, 2.6177151
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944685, 2.1931820
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8632050, 1.8583035
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905749, 1.8948681
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4407959, 2.4391661
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1946788, 2.2031641
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6133065, 2.6011920
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 13.10 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2622

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1779

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3253273, upper bound: 1.3355868
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3351888, upper bound: 1.3268593
time: 3.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3100080, 2.3221502
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256399, 2.6177127
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1943679, 2.1932087
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8636813, 1.8582072
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905821, 1.8949585
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4407930, 2.4394827
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1947737, 2.2032037
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6132975, 2.6013234
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 13.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2809

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1846

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3458790, upper bound: 1.3639559
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3458790, upper bound: 1.3639559
time: 3.90 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3100147, 2.3221431
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256390, 2.6177137
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1943913, 2.1931856
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8636322, 1.8582559
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905945, 1.8949463
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4407940, 2.4394817
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1947231, 2.2032542
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6132984, 2.6013229
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 13.09 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2803

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3448875, upper bound: 1.3656191
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3448506, upper bound: 1.3656735
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.2951922, 2.3023472
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.5975046, 2.6182151
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1918583, 2.1928995
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8575764, 1.8635244
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8712168, 1.8566813
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4292431, 2.4412916
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1932583, 2.1874433
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6001334, 2.6097753
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1978

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3236513, upper bound: 1.3272501
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3345053, upper bound: 1.3144777
time: 4.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3144822, 2.3100607
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6102891, 2.6256411
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1932864, 2.1930401
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8580995, 1.8637285
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8610458, 1.8905749
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4394846, 2.4305551
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2031641, 2.1847777
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6013312, 2.6121075
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1846

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3346040, upper bound: 1.3338248
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3457210, upper bound: 1.3227082
time: 4.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.2951922, 2.3023472
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.5975046, 2.6182151
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1918583, 2.1928995
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8575764, 1.8635244
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8712168, 1.8566813
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4292431, 2.4412916
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1932583, 2.1874433
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6001334, 2.6097753
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 604

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3398254, upper bound: 1.3231604
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3302529, upper bound: 1.3327971
time: 3.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3144822, 2.3100607
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6102891, 2.6256411
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1932864, 2.1930401
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8580995, 1.8637285
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8610458, 1.8905749
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4394846, 2.4305551
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2031641, 2.1847777
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6013312, 2.6121075
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2250

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2482

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1741

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3449339, upper bound: 1.3131935
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3245697, upper bound: 1.3339587
time: 4.54 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 27.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3195125, upper bound: 1.3240874
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3243982, upper bound: 1.3192040
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3253273, upper bound: 1.3355868
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3351888, upper bound: 1.3268593
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3458790, upper bound: 1.3639559
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3458790, upper bound: 1.3639559
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3448875, upper bound: 1.3656191
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3448506, upper bound: 1.3656735
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3236513, upper bound: 1.3272501
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3345053, upper bound: 1.3144777
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3346040, upper bound: 1.3338248
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3457210, upper bound: 1.3227082
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3398254, upper bound: 1.3231604
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3302529, upper bound: 1.3327971
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3449339, upper bound: 1.3131935
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 27.55
Output dim: 7, lower bound: -1.3245697, upper bound: 1.3339587

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3072815, 2.3217158
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6177135, 2.6115232
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1728320, 2.1724982
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8376718, 1.8378024
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8895230, 1.8939118
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4406171, 2.4393284
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1826754, 2.1975193
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6123085, 2.6003270
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 761

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3070544, upper bound: 1.3118648
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3069343, upper bound: 1.3122091
time: 3.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3095808, 2.3194170
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6194491, 2.6097872
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1736803, 2.1716492
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8432269, 1.8322463
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8895473, 1.8938870
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4406400, 2.4393055
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1890354, 2.1911559
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6123018, 2.6003337
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 227

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1760

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1734

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3215336, upper bound: 1.2973762
time: 3.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2983569, upper bound: 1.3176751
time: 3.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3100710, 2.3222413
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256895, 2.6177521
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944661, 2.1932847
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8637443, 1.8583121
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905783, 1.8949406
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4408116, 2.4395087
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1946855, 2.2031794
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6133137, 2.6013472
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1499

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3125

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3250635, upper bound: 1.3351074
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3248482, upper bound: 1.3353228
time: 3.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3101058, 2.3222065
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256781, 2.6177635
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944671, 2.1932838
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8637371, 1.8583193
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8905764, 1.8949430
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4408197, 2.4395001
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1946988, 2.2031660
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6133223, 2.6013396
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3351888, upper bound: 1.3256095
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3305033, upper bound: 1.3268599
time: 3.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3052664, 2.3159945
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6242537, 2.6165395
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944623, 2.1932740
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8430729, 1.8410683
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8891892, 1.8931372
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4408703, 2.4393399
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1951122, 2.2037411
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6120567, 2.6006527
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.61 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1934

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3432353, upper bound: 1.3613989
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3432093, upper bound: 1.3614648
time: 3.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3038592, 2.3174019
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6244655, 2.6163278
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1944566, 2.1932800
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8464932, 1.8376474
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8887734, 1.8935533
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4406509, 2.4395592
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1952605, 2.2035933
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6126270, 2.6000829
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1236

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3313371, upper bound: 1.3598618
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3417490, upper bound: 1.3493840
time: 4.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3090072, 2.3203955
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6163216, 2.6112247
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1821947, 2.1851013
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8609796, 1.8573208
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8845863, 1.8911412
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4405632, 2.4393568
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1948729, 2.2028503
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6042776, 2.5932698
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.82 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1096

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3125

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3444048, upper bound: 1.3647432
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3442753, upper bound: 1.3651102
time: 3.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3082600, 2.3221960
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6191511, 2.6177151
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1862836, 2.1932862
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8637290, 1.8555546
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8867769, 1.8949389
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4406681, 2.4394846
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1943698, 2.2031641
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6052446, 2.6013312
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 634

### Candidate
type: RSZ, layer: 3, pos: 1779

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3356521, upper bound: 1.3644212
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3443539, upper bound: 1.3545624
time: 3.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3154111, 2.3040111
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6068654, 2.6238084
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1420460, 2.1623442
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8381748, 1.8375521
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8798170, 1.8761213
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4238062, 2.4323611
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1619086, 2.1586225
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5864944, 2.5851426
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3236513, upper bound: 1.3220839
time: 4.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3185116, upper bound: 1.3272501
time: 4.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3161464, 2.3032761
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6158829, 2.6147914
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1611619, 2.1432283
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8321271, 1.8436012
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8804855, 1.8754523
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4310493, 2.4251175
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1671052, 2.1534281
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5731688, 2.5984688
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 13.07 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1499

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2336

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3293754, upper bound: 1.3093913
time: 5.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3294254, upper bound: 1.3093257
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3222065, 2.3101060
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6177635, 2.6256781
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1932836, 2.1944671
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8583193, 1.8637371
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8949432, 1.8905766
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4395003, 2.4408200
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2031660, 2.1946988
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6013393, 2.6133220
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 13.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1858

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3215017, upper bound: 1.3225270
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3228195, upper bound: 1.3212144
time: 4.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3222413, 2.3100712
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6177521, 2.6256895
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1932845, 2.1944661
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8583121, 1.8637443
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8949404, 1.8905787
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4395084, 2.4408114
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2031794, 2.1946855
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6013470, 2.6133139
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1846

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2803

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3441379, upper bound: 1.3210168
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3441155, upper bound: 1.3210474
time: 4.39 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3158193, 2.2958024
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6165743, 2.6241231
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1909847, 2.1905658
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8539395, 1.8656158
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8840413, 1.8764317
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4347467, 2.4335041
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2020936, 2.1933861
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5946450, 2.6043310
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1929

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3311578, upper bound: 1.3074239
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3247320, upper bound: 1.3160484
time: 3.64 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3079376, 2.3036840
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6161971, 2.6245008
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1893840, 2.1921670
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8601909, 1.8593645
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8807960, 1.8796771
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4321923, 2.4360585
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2018666, 2.1936131
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5923562, 2.6066189
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1236

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2889940, upper bound: 1.2903163
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2882597, upper bound: 1.2911444
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3104358, 2.2997813
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6091352, 2.6212816
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1936750, 2.1942484
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8309479, 1.8380723
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8978400, 1.8913817
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4383268, 2.4343984
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2035842, 2.1927004
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6002684, 2.6118369
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 227

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3344413, upper bound: 1.2817380
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3058688, upper bound: 1.3053072
time: 4.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3119168, 2.2983003
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6133556, 2.6170611
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1930656, 2.1948576
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8326473, 1.8363738
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8957462, 1.8934760
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4330873, 2.4396384
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.2011809, 2.1951036
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5998621, 2.6122422
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1949

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3245697, upper bound: 1.3289065
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3194450, upper bound: 1.3339585
time: 4.18 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.95 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3070544, upper bound: 1.3118648
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3069343, upper bound: 1.3122091
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3215336, upper bound: 1.2973762
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.2983569, upper bound: 1.3176751
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3250635, upper bound: 1.3351074
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3248482, upper bound: 1.3353228
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3351888, upper bound: 1.3256095
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3305033, upper bound: 1.3268599
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3432353, upper bound: 1.3613989
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3432093, upper bound: 1.3614648
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3313371, upper bound: 1.3598618
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3417490, upper bound: 1.3493840
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3444048, upper bound: 1.3647432
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3442753, upper bound: 1.3651102
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3356521, upper bound: 1.3644212
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3443539, upper bound: 1.3545624
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3236513, upper bound: 1.3220839
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3185116, upper bound: 1.3272501
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3293754, upper bound: 1.3093913
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3294254, upper bound: 1.3093257
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3215017, upper bound: 1.3225270
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3228195, upper bound: 1.3212144
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3441379, upper bound: 1.3210168
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3441155, upper bound: 1.3210474
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3311578, upper bound: 1.3074239
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3247320, upper bound: 1.3160484
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.2889940, upper bound: 1.2903163
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.2882597, upper bound: 1.2911444
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3344413, upper bound: 1.2817380
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3058688, upper bound: 1.3053072
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3245697, upper bound: 1.3289065
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.95
Output dim: 7, lower bound: -1.3194450, upper bound: 1.3339585

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3067431, 2.3203638
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6302924, 2.6050577
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1919823, 2.1884761
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8498044, 1.8505936
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8820839, 1.8826127
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4140878, 2.4224613
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1760545, 2.1883869
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6123667, 2.5987425
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 921

### Candidate
type: RSZ, layer: 3, pos: 411

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3012125, upper bound: 1.3114568
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3065641, upper bound: 1.3041686
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3082285, 2.3188784
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6129837, 2.6223664
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1896591, 2.1908000
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8560185, 1.8443794
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8782487, 1.8864484
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4237723, 2.4127767
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1799064, 2.1845350
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6107168, 2.6003914
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.72 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2965406, upper bound: 1.3022987
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2937419, upper bound: 1.3027498
time: 3.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3092871, 2.3210135
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6208057, 2.6155589
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1935396, 2.1917145
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8627801, 1.8580036
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8867865, 1.8864310
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4406371, 2.4392128
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1943989, 2.2025332
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6132503, 2.6009691
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.84 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1741

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3145331, upper bound: 1.2812880
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3054504, upper bound: 1.2903641
time: 3.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.3088784, 2.3214226
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6234851, 2.6128793
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1928964, 2.1923573
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8634286, 1.8573546
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8820667, 1.8911510
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4405246, 2.4393253
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1940527, 2.2028794
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.6129441, 2.6012757
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2528

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 170

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2735102, upper bound: 1.2881737
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2689152, upper bound: 1.2931679
time: 3.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.2939391, 2.3097997
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6256671, 2.6171889
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1905241, 2.1897414
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8741999, 1.8672752
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8701224, 1.8732903
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4249291, 2.4193203
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1986704, 2.2074351
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5993056, 2.5867109
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.69 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1760

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2622

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.3189477, upper bound: 1.2920881
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.2885878, upper bound: 1.3262807
time: 4.69 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -2.2976646, 2.3060741
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -3.0224142, 3.0224142
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.6251149, 2.6177411
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -2.1909237, 2.1893418
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.8727007, 1.8687744
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.8689256, 1.8744869
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.4206319, 2.4236176
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -2.1989546, 2.2071509
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.5986857, 2.5873318
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.1105423, 2.1105423

Time for backsubstitution: 12.60 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=2.2286510467529297
rel_dist={7: [-1.3704629636566878, 1.370462874585793]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0582297, upper bound: 1.0634195
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0634198, upper bound: 1.0582292
time: 3.85 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.90 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.90
Output dim: 7, lower bound: -1.0582297, upper bound: 1.0634195
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.90
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

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2336

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0518461, upper bound: 1.0563555
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0511630, upper bound: 1.0570391
time: 4.08 seconds

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

Time for backsubstitution: 12.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 411

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0594344, upper bound: 1.0579603
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0631508, upper bound: 1.0542440
time: 4.16 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.80 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.80
Output dim: 7, lower bound: -1.0518461, upper bound: 1.0563555
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.80
Output dim: 7, lower bound: -1.0511630, upper bound: 1.0570391
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.80
Output dim: 7, lower bound: -1.0594344, upper bound: 1.0579603
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.80
Output dim: 7, lower bound: -1.0631508, upper bound: 1.0542440

## BFS RS instance: RS_RSZ1_RSZ1

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

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 604

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0510286, upper bound: 1.0533110
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0488033, upper bound: 1.0555175
time: 4.17 seconds

## BFS RS instance: RS_RSZ1_RSZ2

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

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 18

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1949

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0471837, upper bound: 1.0536847
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0478085, upper bound: 1.0530631
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9790745, 1.9731007
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8083420, 2.8133817
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3568726, 2.3607640
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9612842, 1.9621310
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6274424, 1.6294422
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6029911, 1.6010485
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0775433, 2.0783134
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9866099, 1.9826756
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3243504, 2.3304076
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0792966, 2.0786991

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1499

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0594344, upper bound: 1.0565392
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0580208, upper bound: 1.0579608
time: 3.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9791684, 1.9730067
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8088903, 2.8128333
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3568010, 2.3608356
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9615397, 1.9618754
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6267300, 1.6301556
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6032305, 1.6008091
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0776577, 2.0781989
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9869151, 1.9823699
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3244209, 2.3303370
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0791135, 2.0788817

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2336

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2236

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0483971, upper bound: 1.0339650
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0404382, upper bound: 1.0397760
time: 3.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 20.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -1.0510286, upper bound: 1.0533110
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -1.0488033, upper bound: 1.0555175
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -1.0471837, upper bound: 1.0536847
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -1.0478085, upper bound: 1.0530631
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -1.0594344, upper bound: 1.0565392
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -1.0580208, upper bound: 1.0579608
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -1.0483971, upper bound: 1.0339650
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 20.22
Output dim: 7, lower bound: -1.0404382, upper bound: 1.0397760

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9647655, 1.9727750
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8130598, 2.8080873
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3547263, 2.3522210
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9629569, 1.9618847
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6211257, 1.6214585
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6011162, 1.6037672
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0785222, 2.0780606
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9844537, 1.9886403
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3304658, 2.3246052
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0711370, 2.0692468

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2528

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2336

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0477358, upper bound: 1.0517382
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0495558, upper bound: 1.0499963
time: 5.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9667072, 1.9708331
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8125792, 2.8085680
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3561835, 2.3507638
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9624758, 1.9623656
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6241708, 1.6184134
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6015849, 1.6032984
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0787163, 2.0778666
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9843998, 1.9886937
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3305926, 2.3244789
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0688324, 2.0715518

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 415

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2809

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0465108, upper bound: 1.0359773
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0292625, upper bound: 1.0532234
time: 3.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9706974, 1.9785004
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8154125, 2.8123770
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3648415, 2.3617225
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9697065, 1.9697227
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6447234, 1.6468129
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6240826, 1.6278105
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0871563, 2.0889907
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9757242, 1.9806085
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3097305, 2.3017018
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0850263, 2.0878129

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1858

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0350317, upper bound: 1.0422322
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0350317, upper bound: 1.0422322
time: 3.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9724326, 1.9767656
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8168688, 2.8109212
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3656855, 2.3608785
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9703140, 1.9691155
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6495252, 1.6420112
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6256285, 1.6262646
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0896463, 2.0865006
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9763680, 1.9799647
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3076897, 2.3037426
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0873981, 2.0854411

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1236

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0394861, upper bound: 1.0497946
time: 5.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0443237, upper bound: 1.0451948
time: 4.64 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9698296, 1.9609654
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8129902, 2.8185115
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3682313, 2.3704319
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9486756, 1.9411438
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6339655, 1.6392474
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5863533, 1.5797522
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0402985, 2.0389800
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9816294, 1.9777007
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3231430, 2.3283796
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0779514, 2.0771480

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2236

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0547621, upper bound: 1.0517914
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0547621, upper bound: 1.0517914
time: 4.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9670329, 1.9637620
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8140202, 2.8174820
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3664689, 2.3721943
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9405527, 1.9492664
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6365352, 1.6366782
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5819340, 1.5841713
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0383244, 2.0409541
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9819407, 1.9773889
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3223925, 2.3291302
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0775623, 2.0775371

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1236

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0498740, upper bound: 1.0545142
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0547037, upper bound: 1.0498180
time: 4.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9495053, 1.9422972
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8063269, 2.8091564
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3397169, 2.3455915
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9458036, 1.9477124
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6126366, 1.6080198
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6021137, 1.6003101
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0760589, 2.0771775
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9836001, 1.9768691
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3209167, 2.3292222
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0686479, 2.0621443

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 18

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0418760, upper bound: 1.0327157
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0471426, upper bound: 1.0275648
time: 3.88 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9463758, 1.9434376
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8042707, 2.8108182
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3407817, 2.3436804
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9471211, 1.9452355
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6053076, 1.6140356
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6024923, 1.5996742
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0762286, 2.0767150
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9811096, 1.9789128
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3229899, 2.3269038
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0625587, 2.0669284

Time for backsubstitution: 12.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1978

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1760

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0354063, upper bound: 1.0349178
time: 4.05 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0354063, upper bound: 1.0349178
time: 4.02 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 20.61 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0477358, upper bound: 1.0517382
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0495558, upper bound: 1.0499963
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0465108, upper bound: 1.0359773
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0292625, upper bound: 1.0532234
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0350317, upper bound: 1.0422322
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0350317, upper bound: 1.0422322
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0394861, upper bound: 1.0497946
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0443237, upper bound: 1.0451948
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0547621, upper bound: 1.0517914
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0547621, upper bound: 1.0517914
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0498740, upper bound: 1.0545142
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0547037, upper bound: 1.0498180
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0418760, upper bound: 1.0327157
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0471426, upper bound: 1.0275648
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0354063, upper bound: 1.0349178
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.61
Output dim: 7, lower bound: -1.0354063, upper bound: 1.0349178

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9721761, 1.9788122
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8133850, 2.8081679
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3594551, 2.3565593
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9654307, 1.9626887
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6277041, 1.6251268
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6031113, 1.6028638
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0793872, 2.0783653
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9837728, 1.9897084
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3305206, 2.3242698
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0757308, 2.0784526

Time for backsubstitution: 12.60 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 3118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0453637, upper bound: 1.0432639
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0389294, upper bound: 1.0497146
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9727445, 1.9782438
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8126593, 2.8088932
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3605227, 2.3554921
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9632802, 1.9648399
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6278396, 1.6249919
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6006818, 1.6052933
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0790210, 2.0787315
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9854684, 1.9880133
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3302574, 2.3245335
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0780377, 2.0761452

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1929

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3118

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0476206, upper bound: 1.0478696
time: 4.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0476270, upper bound: 1.0480422
time: 4.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9812293, 1.9852796
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8169222, 2.8133545
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3602653, 2.3565025
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9638395, 1.9639168
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6195922, 1.6183352
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6017814, 1.6038463
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0989008, 2.0964427
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9870138, 1.9906607
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3236642, 2.3173118
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0768633, 2.0774159

Time for backsubstitution: 12.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1760

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 212

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0465108, upper bound: 1.0358044
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0450760, upper bound: 1.0359769
time: 4.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9792118, 1.9872971
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8178463, 2.8124304
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3604655, 2.3563023
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9645076, 1.9632485
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6210480, 1.6168795
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6016641, 1.6039636
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0970984, 2.0982451
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9864202, 1.9912543
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3232989, 2.3176765
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0770016, 2.0772781

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1978

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1236

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0215427, upper bound: 1.0502053
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0262797, upper bound: 1.0452211
time: 4.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9732294, 1.9792738
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8128719, 2.8086476
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3596773, 2.3565412
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9640369, 1.9629169
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6311517, 1.6286130
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6024871, 1.6039357
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0787711, 2.0782874
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9841976, 1.9879918
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3281546, 2.3234015
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0788894, 2.0796123

Time for backsubstitution: 12.68 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 3125

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1206

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0151158, upper bound: 1.0215854
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0143811, upper bound: 1.0223191
time: 3.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9732060, 1.9792848
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8134956, 2.8083806
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3605042, 2.3569298
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9635081, 1.9637749
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6314807, 1.6284389
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6017537, 1.6045220
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0790238, 2.0781152
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9846301, 1.9884381
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3293896, 2.3246417
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0789800, 2.0793037

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1934

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1846

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0337155, upper bound: 1.0409207
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0337155, upper bound: 1.0409207
time: 4.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9672694, 1.9734769
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8126335, 2.8068089
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3591909, 2.3556895
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9614196, 1.9609289
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6223073, 1.6172056
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6015873, 1.6036861
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0791497, 2.0785062
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9808278, 1.9861350
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3249197, 2.3174946
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0782967, 2.0789957

Time for backsubstitution: 12.75 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1934

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 604

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0348216, upper bound: 1.0409212
time: 3.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0306176, upper bound: 1.0451245
time: 4.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9674096, 1.9733369
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8113003, 2.8081417
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3596525, 2.3552275
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9615202, 1.9608285
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6199183, 1.6195946
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6015043, 1.6037698
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0791616, 2.0784943
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9818945, 1.9850683
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3234825, 2.3189323
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0785813, 2.0787110

Time for backsubstitution: 12.81 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0332686, upper bound: 1.0332719
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0332686, upper bound: 1.0332719
time: 4.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9790893, 1.9727850
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8089542, 2.8133860
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3562646, 2.3604269
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9637747, 1.9641066
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6249776, 1.6296988
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6039181, 1.6015856
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0779581, 2.0776000
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9882545, 1.9842176
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3243971, 2.3305564
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0794249, 2.0787101

Time for backsubstitution: 12.80 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 312

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1236

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0463216, upper bound: 1.0481342
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0509981, upper bound: 1.0434381
time: 4.41 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9788527, 1.9730215
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8088942, 2.8134460
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3564639, 2.3602276
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9635153, 1.9643655
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6269865, 1.6276903
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6037679, 1.6017361
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0769444, 2.0786138
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9884577, 1.9840145
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3245687, 2.3303843
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0791245, 2.0790105

Time for backsubstitution: 12.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1779

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0398016, upper bound: 1.0378768
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0418109, upper bound: 1.0375390
time: 4.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9733372, 1.9674094
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8081417, 2.8113008
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3552275, 2.3596525
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9608283, 1.9615200
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6195946, 1.6199183
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6037703, 1.6015038
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0784941, 2.0791619
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9850683, 1.9818950
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3189325, 2.3234823
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0787110, 2.0785813

Time for backsubstitution: 12.56 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1111

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0449765, upper bound: 1.0494588
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0449765, upper bound: 1.0494588
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9734769, 1.9672692
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8068085, 2.8126335
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3556895, 2.3591909
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9609289, 1.9614196
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6172056, 1.6223073
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6036863, 1.6015878
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0785060, 2.0791500
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9861350, 1.9808278
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3174944, 2.3249195
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0789957, 2.0782962

Time for backsubstitution: 12.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2809

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2146

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0411008, upper bound: 1.0386206
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0434819, upper bound: 1.0362385
time: 4.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9738922, 1.9674981
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8095675, 2.8136249
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3562655, 2.3605981
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9639506, 1.9646723
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6253161, 1.6266789
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6055856, 1.6033540
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0951791, 2.1000590
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9845924, 1.9811630
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3255563, 2.3310061
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0789909, 2.0786562

Time for backsubstitution: 12.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2482

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0380079, upper bound: 1.0327077
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0418671, upper bound: 1.0287675
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9735656, 1.9678247
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8091335, 2.8140597
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3566351, 2.3602281
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9640813, 1.9645417
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6239662, 1.6280289
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6055360, 1.6034033
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0994034, 2.0958347
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9854031, 1.9803524
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3250184, 2.3315444
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0790706, 2.0785761

Time for backsubstitution: 12.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 634

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2482

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 901

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0421404, upper bound: 1.0252671
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0456260, upper bound: 1.0245091
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9788051, 1.9738750
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8083839, 2.8144207
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3567986, 2.3606319
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9632282, 1.9649591
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6293926, 1.6310778
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6036806, 1.6024621
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0775537, 2.0791156
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9883819, 1.9844666
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3238611, 2.3304238
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0800233, 2.0787930

Time for backsubstitution: 12.67 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2236

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 411

### Candidate
type: RSZ, layer: 3, pos: 3125

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0347573, upper bound: 1.0342336
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0317832, upper bound: 1.0342794
time: 4.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9792848, 1.9727373
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8090048, 2.8128753
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3569298, 2.3607616
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9637752, 1.9638190
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6283655, 1.6314802
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.6045222, 1.6014984
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0783682, 2.0782096
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9887066, 1.9846306
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.3244371, 2.3306293
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0792079, 2.0789800

Time for backsubstitution: 12.71 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 212

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0354063, upper bound: 1.0338056
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0342402, upper bound: 1.0349166
time: 4.32 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 21.03 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0453637, upper bound: 1.0432639
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0389294, upper bound: 1.0497146
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0476206, upper bound: 1.0478696
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0476270, upper bound: 1.0480422
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0465108, upper bound: 1.0358044
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0450760, upper bound: 1.0359769
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0215427, upper bound: 1.0502053
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0262797, upper bound: 1.0452211
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0151158, upper bound: 1.0215854
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0143811, upper bound: 1.0223191
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0337155, upper bound: 1.0409207
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0337155, upper bound: 1.0409207
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0348216, upper bound: 1.0409212
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0306176, upper bound: 1.0451245
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0332686, upper bound: 1.0332719
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0332686, upper bound: 1.0332719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0463216, upper bound: 1.0481342
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0509981, upper bound: 1.0434381
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0398016, upper bound: 1.0378768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0418109, upper bound: 1.0375390
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0449765, upper bound: 1.0494588
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0449765, upper bound: 1.0494588
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0411008, upper bound: 1.0386206
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0434819, upper bound: 1.0362385
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0380079, upper bound: 1.0327077
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0418671, upper bound: 1.0287675
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0421404, upper bound: 1.0252671
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0456260, upper bound: 1.0245091
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0347573, upper bound: 1.0342336
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0317832, upper bound: 1.0342794
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0354063, upper bound: 1.0338056
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 21.03
Output dim: 7, lower bound: -1.0342402, upper bound: 1.0349166

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

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1096

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1779

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0413013, upper bound: 1.0432535
time: 5.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0453552, upper bound: 1.0391917
time: 4.36 seconds

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

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 921

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0269549, upper bound: 1.0382764
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0282731, upper bound: 1.0350323
time: 4.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9514518, 1.9551003
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8086138, 2.8077049
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3541012, 2.3475161
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9678702, 1.9664576
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6139607, 1.6098409
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5871572, 1.5897760
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0463037, 2.0470638
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9825034, 1.9870591
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2920551, 2.2854097
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0735641, 2.0743461

Time for backsubstitution: 15.05 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 604

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Candidate
type: RSZ, layer: 3, pos: 2250

### Candidate
type: RSZ, layer: 3, pos: 2803

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0462217, upper bound: 1.0465156
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0462070, upper bound: 1.0465299
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.9492168, 1.9575195
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.8121967, 2.8043003
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.3515635, 2.3501377
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.9670491, 1.9673202
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.6125531, 1.6113753
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.5876741, 1.5893393
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -2.0479631, 2.0456486
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.9828649, 1.9867439
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.2915993, 2.2860677
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -2.0739317, 2.0740080

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1779

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0435594, upper bound: 1.0480321
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -1.0476188, upper bound: 1.0439727
time: 4.61 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 24.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.01
Output dim: 7, lower bound: -1.0413013, upper bound: 1.0432535
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.01
Output dim: 7, lower bound: -1.0453552, upper bound: 1.0391917
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.01
Output dim: 7, lower bound: -1.0269549, upper bound: 1.0382764
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.01
Output dim: 7, lower bound: -1.0282731, upper bound: 1.0350323
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.01
Output dim: 7, lower bound: -1.0462217, upper bound: 1.0465156
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.01
Output dim: 7, lower bound: -1.0462070, upper bound: 1.0465299
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 24.01
Output dim: 7, lower bound: -1.0435594, upper bound: 1.0480321
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 24.01
Output dim: 7, lower bound: -1.0476188, upper bound: 1.0439727
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0465108, upper bound: 1.0358044
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0450760, upper bound: 1.0359769
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0215427, upper bound: 1.0502053
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0262797, upper bound: 1.0452211
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0151158, upper bound: 1.0215854
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0143811, upper bound: 1.0223191
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0337155, upper bound: 1.0409207
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0337155, upper bound: 1.0409207
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0348216, upper bound: 1.0409212
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0306176, upper bound: 1.0451245
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0332686, upper bound: 1.0332719
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0332686, upper bound: 1.0332719
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0463216, upper bound: 1.0481342
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0509981, upper bound: 1.0434381
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0398016, upper bound: 1.0378768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0418109, upper bound: 1.0375390
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0449765, upper bound: 1.0494588
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0449765, upper bound: 1.0494588
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0411008, upper bound: 1.0386206
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0434819, upper bound: 1.0362385
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0380079, upper bound: 1.0327077
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0418671, upper bound: 1.0287675
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0421404, upper bound: 1.0252671
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0456260, upper bound: 1.0245091
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0347573, upper bound: 1.0342336
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0317832, upper bound: 1.0342794
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0354063, upper bound: 1.0338056
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 24.01
Output dim: 7, lower bound: -1.0342402, upper bound: 1.0349166
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=2.0185980796813965
rel_dist={7: [-1.0634246550232, 1.0634268200891506]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6181

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7908454, upper bound: 0.7927002
time: 9.30 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7927021, upper bound: 0.7908454
time: 4.25 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.57 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.57
Output dim: 7, lower bound: -0.7908454, upper bound: 0.7927002
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.57
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

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2236

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 676

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7872736, upper bound: 0.7891285
time: 5.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7872736, upper bound: 0.7923196
time: 7.96 seconds

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

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1741

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1236

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7903081, upper bound: 0.7902127
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7920710, upper bound: 0.7884527
time: 4.73 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 23.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.98
Output dim: 7, lower bound: -0.7872736, upper bound: 0.7891285
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.98
Output dim: 7, lower bound: -0.7872736, upper bound: 0.7923196
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 23.98
Output dim: 7, lower bound: -0.7903081, upper bound: 0.7902127
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 23.98
Output dim: 7, lower bound: -0.7920710, upper bound: 0.7884527

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7260189, 1.7234387
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.5845890, 2.5815396
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1786394, 2.1760335
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7860279, 1.7852669
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.3518829, 1.3457282
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.3899245, 1.3902533
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8332109, 1.8319957
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8003812, 1.8014307
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.0994544, 2.0963285
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9438090, 1.9383078

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2622

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2153

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7899332, upper bound: 0.7875088
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7882503, upper bound: 0.7882646
time: 7.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7214160, 1.7280416
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.5830364, 2.5830913
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1773548, 2.1773186
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7854638, 1.7858305
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.3466320, 1.3509789
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.3895259, 1.3906519
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8322144, 1.8329928
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8000174, 1.8017945
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.0983243, 2.0974586
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9381695, 1.9439473

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 430

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7839350, upper bound: 0.7891288
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7839338, upper bound: 0.7890400
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7448230, 1.7428470
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6151590, 2.6162124
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1816788, 2.1831536
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8078880, 1.8081186
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4665709, 1.4666786
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4101028, 1.4093473
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8377576, 1.8379805
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8429170, 1.8418593
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1335144, 2.1350312
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9851127, 1.9850693

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1499

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7884316, upper bound: 0.7884533
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7885398, upper bound: 0.7883626
time: 4.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7448697, 1.7428002
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6147156, 2.6166563
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1818328, 2.1829996
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8079214, 1.8080852
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4657745, 1.4674749
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4100752, 1.4093752
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8377619, 1.8379762
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8432727, 1.8415036
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1330357, 2.1355100
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9852076, 1.9849744

Time for backsubstitution: 14.62 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1206

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1779

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7907900, upper bound: 0.7884453
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7920679, upper bound: 0.7871681
time: 4.83 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 24.28 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 7, lower bound: -0.7899332, upper bound: 0.7875088
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 7, lower bound: -0.7882503, upper bound: 0.7882646
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 7, lower bound: -0.7839350, upper bound: 0.7891288
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 7, lower bound: -0.7839338, upper bound: 0.7890400
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 7, lower bound: -0.7884316, upper bound: 0.7884533
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 7, lower bound: -0.7885398, upper bound: 0.7883626
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 7, lower bound: -0.7907900, upper bound: 0.7884453
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 24.28
Output dim: 7, lower bound: -0.7920679, upper bound: 0.7871681

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7315836, 1.7319174
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6071930, 2.6054277
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1824694, 2.1805882
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7922640, 1.7924268
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4726439, 1.4707146
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.3865757, 1.3884006
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8209982, 1.8198333
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8421917, 1.8429642
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1402869, 2.1388016
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9782686, 1.9747114

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2528

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7806007, upper bound: 0.7816329
time: 5.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7840800, upper bound: 0.7801136
time: 6.54 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7298951, 1.7336059
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6069260, 2.6056952
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1819091, 2.1811481
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7926240, 1.7920668
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4716187, 1.4717393
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.3876734, 1.3873034
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8200512, 1.8207798
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8415508, 1.8436050
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1407981, 2.1382904
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9745731, 1.9784069

Time for backsubstitution: 14.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 415

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 634

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7882377, upper bound: 0.7879854
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7881810, upper bound: 0.7882560
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7486515, 1.7506666
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6168957, 2.6154871
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1837296, 2.1826844
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8102827, 1.8099096
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4763198, 1.4754734
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4098415, 1.4103246
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8375893, 1.8374283
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8441620, 1.8454270
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1405258, 2.1389418
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9853773, 1.9856181

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1741

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 634

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7839212, upper bound: 0.7889057
time: 5.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7837140, upper bound: 0.7891151
time: 5.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7486439, 1.7506776
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6175194, 2.6153984
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1840053, 2.1830730
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8101068, 1.8107674
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4766479, 1.4754152
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4095969, 1.4109106
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8378420, 1.8373706
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8445950, 1.8455758
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1409378, 2.1401820
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9854679, 1.9855156

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 921

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2236

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7800339, upper bound: 0.7800837
time: 5.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7774018, upper bound: 0.7852598
time: 5.82 seconds

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

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 430

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 95

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7861285, upper bound: 0.7880166
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7879963, upper bound: 0.7880022
time: 4.74 seconds

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

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1846

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7884031, upper bound: 0.7881346
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7883165, upper bound: 0.7882273
time: 4.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7506876, 1.7486711
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6160297, 2.6175265
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1831212, 2.1844401
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8107653, 1.8109624
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4757538, 1.4766569
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4109144, 1.4101870
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8376389, 1.8378589
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8460212, 1.8446102
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1401901, 2.1421881
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9855680, 1.9854283

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1206

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1111

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7898609, upper bound: 0.7873818
time: 5.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7897439, upper bound: 0.7874989
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7506933, 1.7486653
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6160297, 2.6175275
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1831193, 2.1844420
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8107653, 1.8109624
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4757528, 1.4766579
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4109144, 1.4101872
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8376403, 1.8378575
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8460236, 1.8446078
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1401920, 2.1421862
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9855666, 1.9854298

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1096

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3125

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7910188, upper bound: 0.7868391
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7917424, upper bound: 0.7861252
time: 4.78 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.68 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7806007, upper bound: 0.7816329
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7840800, upper bound: 0.7801136
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7882377, upper bound: 0.7879854
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7881810, upper bound: 0.7882560
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7839212, upper bound: 0.7889057
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7837140, upper bound: 0.7891151
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7800339, upper bound: 0.7800837
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7774018, upper bound: 0.7852598
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7861285, upper bound: 0.7880166
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7879963, upper bound: 0.7880022
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7884031, upper bound: 0.7881346
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7883165, upper bound: 0.7882273
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7898609, upper bound: 0.7873818
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7897439, upper bound: 0.7874989
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7910188, upper bound: 0.7868391
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.68
Output dim: 7, lower bound: -0.7917424, upper bound: 0.7861252

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7377262, 1.7429638
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.5993528, 2.6021194
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1748371, 2.1756468
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8095369, 1.8093159
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4763570, 1.4755402
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.3779850, 1.3770173
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8276010, 1.8291719
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8346891, 1.8365469
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1409793, 2.1385951
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9842186, 1.9835038

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1760

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 2153

### Candidate
type: RSZ, layer: 3, pos: 1934

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7583079, upper bound: 0.7589859
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7583079, upper bound: 0.7589854
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7409410, 1.7397490
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6036167, 2.5978556
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1769676, 2.1735158
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8095131, 1.8093393
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4764442, 1.4754529
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.3762898, 1.3787124
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8293905, 1.8273823
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8351336, 1.8361030
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1405911, 2.1389832
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9833660, 1.9843569

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2528

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7837953, upper bound: 0.7770575
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7809572, upper bound: 0.7798302
time: 5.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7486019, 1.7506256
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6175470, 2.6160479
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1843925, 2.1830711
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8108640, 1.8106709
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4765601, 1.4756474
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4102006, 1.4109302
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8378401, 1.8376217
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8446851, 1.8460903
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1421700, 2.1401744
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9854221, 1.9855599

Time for backsubstitution: 14.39 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2153

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 761

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7859833, upper bound: 0.7855015
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7857583, upper bound: 0.7857253
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7486033, 1.7506247
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6175451, 2.6160507
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1843920, 2.1830711
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8108678, 1.8106670
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4765515, 1.4756556
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4102030, 1.4109282
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8378401, 1.8376212
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8446765, 1.8460984
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1421700, 2.1401744
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9854217, 1.9855604

Time for backsubstitution: 14.38 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 227

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2146

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7796046, upper bound: 0.7822872
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7824129, upper bound: 0.7806424
time: 5.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7486019, 1.7506256
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6175470, 2.6160479
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1843925, 2.1830711
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8108640, 1.8106709
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4765601, 1.4756474
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4102006, 1.4109302
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8378401, 1.8376217
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8446851, 1.8460903
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1421700, 2.1401744
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9854221, 1.9855599

Time for backsubstitution: 14.40 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1111

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 312

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7827557, upper bound: 0.7881572
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7831904, upper bound: 0.7877120
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7486033, 1.7506247
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6175451, 2.6160507
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1843920, 2.1830711
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8108678, 1.8106670
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4765515, 1.4756556
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4102030, 1.4109282
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8378401, 1.8376212
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8446765, 1.8460984
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1421700, 2.1401744
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9854217, 1.9855604

Time for backsubstitution: 14.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1499

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 921

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1934

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7610364, upper bound: 0.7663753
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7610205, upper bound: 0.7663930
time: 4.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7184954, 1.7208977
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6142874, 2.6133442
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1678185, 2.1658602
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7943110, 1.7936747
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4531879, 1.4547262
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4081535, 1.4087548
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8356876, 1.8353148
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8368340, 1.8390775
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1392255, 2.1364570
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9686322, 1.9707999

Time for backsubstitution: 14.37 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2809

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1499

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7773701, upper bound: 0.7852364
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7773701, upper bound: 0.7852239
time: 5.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7541170, 1.7530322
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6137495, 2.6155205
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1875896, 2.1892557
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7971134, 1.7981672
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4693651, 1.4704361
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4018898, 1.4018033
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8323426, 1.8336082
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8423114, 1.8408837
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1129489, 2.1144435
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9838572, 1.9838533

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1741

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1978

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7836771, upper bound: 0.7864810
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7862908, upper bound: 0.7850637
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7550545, 1.7520947
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6140242, 2.6152472
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1879344, 2.1889105
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7979703, 1.7973101
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4695320, 1.4702692
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4025307, 1.4011624
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8333893, 1.8325615
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8422971, 1.8408980
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1124473, 2.1149452
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9839916, 1.9837189

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 921

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2153

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7871584, upper bound: 0.7856937
time: 4.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7838090, upper bound: 0.7871629
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7458830, 1.7436261
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6142044, 2.6156230
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1818619, 2.1832180
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8107615, 1.8109574
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4579387, 1.4594131
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4095244, 1.4087281
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8376980, 1.8378799
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8464375, 1.8450489
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1389341, 2.1410246
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9891610, 1.9890342

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2336

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7873738, upper bound: 0.7880582
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7883120, upper bound: 0.7870984
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7456484, 1.7438607
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6141262, 2.6157017
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1818972, 2.1831827
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8107605, 1.8109586
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4585090, 1.4588428
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4094558, 1.4087975
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8376613, 1.8379166
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8464618, 1.8450241
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1390285, 2.1409297
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9891720, 1.9890227

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 430

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7849686, upper bound: 0.7849141
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7849686, upper bound: 0.7849141
time: 5.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7504816, 1.7483802
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6159716, 2.6174493
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1825404, 2.1839280
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8107672, 1.8108778
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4732928, 1.4748669
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4103069, 1.4095292
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8372135, 1.8370943
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8455276, 1.8441820
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1400518, 2.1421051
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9854364, 1.9851980

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 212

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1858

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7892150, upper bound: 0.7862888
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7887895, upper bound: 0.7867127
time: 4.84 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7504029, 1.7484589
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6159515, 2.6174693
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1826067, 2.1838613
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8106804, 1.8109641
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4739628, 1.4741974
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4102569, 1.4095795
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8368754, 1.8374319
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8455954, 1.8441143
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1401091, 2.1420479
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9853363, 1.9852982

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 3125
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 604

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1929

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7876822, upper bound: 0.7854129
time: 4.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7877126, upper bound: 0.7853890
time: 5.07 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7376604, 1.7362585
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6079969, 2.6094394
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1826386, 2.1838675
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8071561, 1.8074198
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4849658, 1.4856200
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.3894610, 1.3885343
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8181753, 1.8176780
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8499956, 1.8486295
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1256652, 2.1275578
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9856677, 1.9855275

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 2622
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 2336

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 415

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7891566, upper bound: 0.7846365
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7888333, upper bound: 0.7849533
time: 4.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7382812, 1.7356377
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6079416, 2.6094947
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1825466, 2.1839595
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.8072228, 1.8073530
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4847159, 1.4858704
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.3892617, 1.3887339
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.8174591, 1.8183942
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8500428, 1.8485818
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1255622, 2.1276608
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9856658, 1.9855299

Time for backsubstitution: 14.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 921
type: RSZ, layer: 3, pos: 604
type: RSZ, layer: 3, pos: 1846
type: RSZ, layer: 3, pos: 1858
type: RSZ, layer: 3, pos: 1779
type: RSZ, layer: 3, pos: 1741
type: RSZ, layer: 3, pos: 1949
type: RSZ, layer: 3, pos: 761
type: RSZ, layer: 3, pos: 1206
type: RSZ, layer: 3, pos: 170
type: RSZ, layer: 3, pos: 2482
type: RSZ, layer: 3, pos: 2236
type: RSZ, layer: 3, pos: 2153
type: RSZ, layer: 3, pos: 219
type: RSZ, layer: 3, pos: 2336
type: RSZ, layer: 3, pos: 676
type: RSZ, layer: 3, pos: 2146
type: RSZ, layer: 3, pos: 227
type: RSZ, layer: 3, pos: 2803
type: RSZ, layer: 3, pos: 1236
type: RSZ, layer: 3, pos: 634
type: RSZ, layer: 3, pos: 2809
type: RSZ, layer: 3, pos: 1734
type: RSZ, layer: 3, pos: 1978
type: RSZ, layer: 3, pos: 1111
type: RSZ, layer: 3, pos: 1929
type: RSZ, layer: 3, pos: 1096
type: RSZ, layer: 3, pos: 2250
type: RSZ, layer: 3, pos: 3118
type: RSZ, layer: 3, pos: 1760
type: RSZ, layer: 3, pos: 901
type: RSZ, layer: 3, pos: 95
type: RSZ, layer: 3, pos: 411
type: RSZ, layer: 3, pos: 2528
type: RSZ, layer: 3, pos: 430
type: RSZ, layer: 3, pos: 1934
type: RSZ, layer: 3, pos: 415
type: RSZ, layer: 3, pos: 312
type: RSZ, layer: 3, pos: 18
type: RSZ, layer: 3, pos: 212
type: RSZ, layer: 3, pos: 1499
type: RSZ, layer: 3, pos: 2622

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 921

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7887085, upper bound: 0.7833442
time: 5.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7887085, upper bound: 0.7833442
time: 5.73 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 25.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7583079, upper bound: 0.7589859
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7583079, upper bound: 0.7589854
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7837953, upper bound: 0.7770575
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7809572, upper bound: 0.7798302
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7859833, upper bound: 0.7855015
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7857583, upper bound: 0.7857253
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7796046, upper bound: 0.7822872
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7824129, upper bound: 0.7806424
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7827557, upper bound: 0.7881572
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7831904, upper bound: 0.7877120
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7610364, upper bound: 0.7663753
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7610205, upper bound: 0.7663930
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7773701, upper bound: 0.7852364
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7773701, upper bound: 0.7852239
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7836771, upper bound: 0.7864810
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7862908, upper bound: 0.7850637
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7871584, upper bound: 0.7856937
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7838090, upper bound: 0.7871629
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7873738, upper bound: 0.7880582
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7883120, upper bound: 0.7870984
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7849686, upper bound: 0.7849141
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7849686, upper bound: 0.7849141
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7892150, upper bound: 0.7862888
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7887895, upper bound: 0.7867127
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7876822, upper bound: 0.7854129
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7877126, upper bound: 0.7853890
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7891566, upper bound: 0.7846365
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7888333, upper bound: 0.7849533
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7887085, upper bound: 0.7833442
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 25.94
Output dim: 7, lower bound: -0.7887085, upper bound: 0.7833442

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -8.1700583, -5.3680067, -8.1700583, -5.3680067, -1.7246842, 1.7259049
1: -9.2260056, -6.2035913, -9.2260056, -6.2035913, -2.6031017, 2.6022620
2: -9.9503212, -6.9502754, -9.9503212, -6.9502754, -2.1865883, 2.1849113
3: -10.8334827, -8.2661476, -10.8334827, -8.2661476, -1.7963438, 1.7952061
4: -5.5582318, -3.5118723, -5.5582318, -3.5118723, -1.4705844, 1.4697018
5: -8.8875761, -6.1918221, -8.8875761, -6.1918221, -1.4107399, 1.4115233
6: -12.9723425, -9.7499943, -12.9723425, -9.7499943, -1.7989869, 1.7983851
7: 0.4052801, 2.8421252, 0.4052801, 2.8421252, -1.8451581, 1.8464856
8: -3.7202172, -0.9862285, -3.7202172, -0.9862285, -2.1445880, 2.1427784
9: 0.1555150, 2.2660573, 0.1555150, 2.2660573, -1.9773808, 1.9769192

Time for backsubstitution: 14.49 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.8785624504089355
rel_dist={7: [-0.7927072900922123, 0.7927031607168487]}

## Binary Search with RS_random_Z Result
status: None
Maximum delta epsilon: None
execution time: 2423.20 seconds
