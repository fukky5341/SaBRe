## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 1.49125804913
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699)
1: (-13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2968750, 4.2968750)
2: (-7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.4000723, 3.4000723)
3: (-12.8481083, -9.6415062, -12.8481083, -9.6415062, -3.2066021, 3.2066021)
4: (-6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.5599658, 3.5599658)
5: (-2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392)
6: (8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471)
7: (-18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.5956373, 3.5956373)
8: (-1.4227927, 1.5777073, -1.4227927, 1.5777073, -3.0005000, 3.0005000)
9: (-16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6913013, 3.6913013)

## BASE Result
execution time: IAR + LP analysis = 15.18 + 32.38 = 47.56 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.44 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=3.453947067260742
rel_dist={6: [-1.951663254659758, 1.9516631580018462]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=3.1755638122558594
rel_dist={6: [-1.493016446904722, 1.493015957466044]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=2.9873008728027344
rel_dist={6: [-1.11031708800226, 1.1103168328917867]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=3.081432342529297
rel_dist={6: [-1.3094574363629174, 1.309455504183358]}

## Binary Search Result
Binary search time: 209.65 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3342.79 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 803

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0958718, upper bound: 2.0860927
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0860928, upper bound: 2.0958717
time: 7.43 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.25 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.25
Output dim: 6, lower bound: -2.0958718, upper bound: 2.0860927
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.25
Output dim: 6, lower bound: -2.0860928, upper bound: 2.0958717

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2531662, 4.2553263
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1831851, 3.1799026
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9644337, 2.9647379
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3771095, 3.3768892
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1833105, 3.1814799
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9442048, 2.9488010
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6225815, 3.6226673

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 819

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 482

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0910026, upper bound: 2.0860879
time: 7.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0958667, upper bound: 2.0812233
time: 9.67 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2553253, 4.2531652
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1799026, 3.1831851
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9647379, 2.9644337
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3768892, 3.3771095
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1814795, 3.1833105
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9488006, 2.9442053
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6226673, 3.6225815

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4645

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0860805, upper bound: 2.0950505
time: 6.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0852667, upper bound: 2.0958598
time: 6.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.71 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.71
Output dim: 6, lower bound: -2.0910026, upper bound: 2.0860879
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.71
Output dim: 6, lower bound: -2.0958667, upper bound: 2.0812233
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.71
Output dim: 6, lower bound: -2.0860805, upper bound: 2.0950505
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.71
Output dim: 6, lower bound: -2.0852667, upper bound: 2.0958598

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2320557, 4.2255039
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1819720, 3.1822848
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9644842, 2.9663424
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3765154, 3.3753815
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1805568, 3.1775923
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9479642, 2.9513350
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6100121, 3.6005983

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 819

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0845964, upper bound: 2.0860817
time: 9.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0909964, upper bound: 2.0798939
time: 13.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2233429, 4.2342157
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1855674, 3.1786895
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9660387, 2.9647884
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3756018, 3.3762946
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1794238, 3.1787262
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9467397, 2.9525599
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6005135, 3.6100960

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 4554

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0958535, upper bound: 2.0620601
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0767008, upper bound: 2.0812098
time: 7.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2571850, 4.2521935
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1815896, 3.1823077
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9682665, 2.9625807
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3775063, 3.3767805
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1814060, 3.1834483
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9480271, 2.9456830
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6212139, 3.6253414

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 6185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0806887, upper bound: 2.0798476
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0806582, upper bound: 2.0950405
time: 6.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2543526, 4.2531652
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1790252, 3.1831851
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9628849, 2.9644337
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3765602, 3.3771095
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1814795, 3.1832366
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9488006, 2.9434319
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6226673, 3.6211300

Time for backsubstitution: 14.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6199

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0852588, upper bound: 2.0859102
time: 5.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0753166, upper bound: 2.0958545
time: 6.10 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 26.29 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 6, lower bound: -2.0845964, upper bound: 2.0860817
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 6, lower bound: -2.0909964, upper bound: 2.0798939
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 6, lower bound: -2.0958535, upper bound: 2.0620601
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 6, lower bound: -2.0767008, upper bound: 2.0812098
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 6, lower bound: -2.0806887, upper bound: 2.0798476
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 6, lower bound: -2.0806582, upper bound: 2.0950405
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 6, lower bound: -2.0852588, upper bound: 2.0859102
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 26.29
Output dim: 6, lower bound: -2.0753166, upper bound: 2.0958545

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2618675, 4.2693882
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1522460, 3.1403356
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9774828, 2.9854708
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3386822, 3.3219786
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1650286, 3.1556983
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9139175, 2.9272118
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5893259, 3.5713997

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4554

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0826561, upper bound: 2.0860355
time: 6.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0845476, upper bound: 2.0841165
time: 5.73 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2759399, 4.2553148
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1400228, 3.1525583
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9836130, 2.9793420
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3231115, 3.3375483
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1586618, 3.1620641
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9238415, 2.9172878
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5808125, 3.5799141

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5717

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4645

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0909842, upper bound: 2.0790713
time: 6.51 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0901719, upper bound: 2.0798816
time: 9.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2209158, 4.2323322
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1855707, 3.1786947
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9659061, 2.9646173
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3715324, 3.3731422
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1748738, 3.1755071
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9572420, 2.9605751
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5931129, 3.6048574

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 6185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 110

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0957609, upper bound: 2.0551555
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0889358, upper bound: 2.0620604
time: 6.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2214613, 4.2317858
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1855726, 3.1786933
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9658670, 2.9646559
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3724499, 3.3722258
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1762052, 3.1741776
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9547548, 2.9630637
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5952740, 3.6026955

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0691621, upper bound: 2.0799068
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0765708, upper bound: 2.0798123
time: 8.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2438984, 4.2466602
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1644397, 3.1549025
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9685488, 2.9630637
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3743801, 3.3741217
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1777830, 3.1732607
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9238005, 2.9333572
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6266451, 3.6293001

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6199

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0806818, upper bound: 2.0698975
time: 9.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0707373, upper bound: 2.0798400
time: 6.97 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2516613, 4.2389078
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1541848, 3.1651607
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9687500, 2.9632559
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3752871, 3.3736548
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1712179, 3.1798406
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9357004, 2.9214559
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6251745, 3.6309423

Time for backsubstitution: 14.48 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 4554

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 515

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0806415, upper bound: 2.0812964
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0669148, upper bound: 2.0950240
time: 5.18 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2569981, 4.2571449
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1815014, 3.1928144
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9465828, 2.9386163
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3836689, 3.3898582
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1815672, 3.1833673
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9442959, 2.9402390
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5766964, 3.5885382

Time for backsubstitution: 14.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0808591, upper bound: 2.0857734
time: 14.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0851244, upper bound: 2.0815082
time: 13.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2583332, 4.2558117
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1886539, 3.1856618
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9370670, 2.9481316
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3893099, 3.3842173
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1816092, 3.1833253
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9456072, 2.9389272
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5900764, 3.5751591

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 4554

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0691261, upper bound: 2.0958472
time: 5.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0753106, upper bound: 2.0894491
time: 7.18 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 26.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0826561, upper bound: 2.0860355
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0845476, upper bound: 2.0841165
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0909842, upper bound: 2.0790713
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0901719, upper bound: 2.0798816
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0957609, upper bound: 2.0551555
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0889358, upper bound: 2.0620604
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0691621, upper bound: 2.0799068
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0765708, upper bound: 2.0798123
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0806818, upper bound: 2.0698975
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0707373, upper bound: 2.0798400
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0806415, upper bound: 2.0812964
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0669148, upper bound: 2.0950240
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0808591, upper bound: 2.0857734
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0851244, upper bound: 2.0815082
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0691261, upper bound: 2.0958472
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 26.89
Output dim: 6, lower bound: -2.0753106, upper bound: 2.0894491

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2618217, 4.2695065
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1521497, 3.1405797
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9780750, 2.9852405
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3397198, 3.3215742
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1639595, 3.1584611
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9150386, 2.9267745
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5887995, 3.5728140

Time for backsubstitution: 14.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 110

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0824880, upper bound: 2.0791287
time: 7.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0757168, upper bound: 2.0860338
time: 5.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2618675, 4.2693434
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1522460, 3.1402397
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9772530, 2.9854708
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3382778, 3.3219786
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1650286, 3.1546302
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9134803, 2.9272118
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5893259, 3.5708733

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 4645

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 515

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0845304, upper bound: 2.0703740
time: 7.23 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0709764, upper bound: 2.0840992
time: 6.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2777977, 4.2543421
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1417089, 3.1516800
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9871407, 2.9774876
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3237305, 3.3372216
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1585903, 3.1622028
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9230671, 2.9187660
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5793600, 3.5826731

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0909696, upper bound: 2.0456597
time: 7.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0575993, upper bound: 2.0790542
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2749672, 4.2553148
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1391444, 3.1525583
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9817581, 2.9793420
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3227854, 3.3375483
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1586618, 3.1619911
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9238415, 2.9165144
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5808125, 3.5784607

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 6185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6199

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0901639, upper bound: 2.0699300
time: 7.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0802217, upper bound: 2.0798748
time: 32.19 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2209158, 4.2323313
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1855717, 3.1786957
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9659066, 2.9646177
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3715334, 3.3731413
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1748738, 3.1755061
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9572420, 2.9605751
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5931158, 3.6048574

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 4645

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0914579, upper bound: 2.0551526
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0957581, upper bound: 2.0508198
time: 5.31 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2209139, 4.2323332
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1855717, 3.1786957
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9659066, 2.9646187
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3715315, 3.3731427
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1748738, 3.1755066
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9572420, 2.9605742
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5931139, 3.6048613

Time for backsubstitution: 14.49 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0889280, upper bound: 2.0605750
time: 6.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0874544, upper bound: 2.0620529
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2212381, 4.2312527
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1793909, 3.1747561
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9658518, 2.9650292
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3723297, 3.3719444
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1749120, 3.1711874
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9554448, 2.9630537
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5966396, 3.6026793

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 110

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0690714, upper bound: 2.0730017
time: 5.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0622452, upper bound: 2.0799067
time: 9.60 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2209272, 4.2315636
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1816349, 3.1769924
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9662437, 2.9646401
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3721676, 3.3721094
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1732144, 3.1728849
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9547429, 2.9637542
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5952587, 3.6040602

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 6199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0765626, upper bound: 2.0743916
time: 14.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0613735, upper bound: 2.0744200
time: 7.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2465487, 4.2506437
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1669178, 3.1645322
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9522371, 2.9372458
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3814859, 3.3868685
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1778698, 3.1733894
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9192958, 2.9301624
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5806761, 3.5967197

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 110

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0806818, upper bound: 2.0630454
time: 4.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0737581, upper bound: 2.0698976
time: 6.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2478819, 4.2493105
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1740713, 3.1573801
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9427309, 2.9467616
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3871212, 3.3812275
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1779118, 3.1733484
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9206071, 2.9288530
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5940561, 3.5833321

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 515

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 805

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0707202, upper bound: 2.0463945
time: 7.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0372938, upper bound: 2.0798272
time: 6.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2570696, 4.2580767
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1649494, 3.1845984
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9875889, 2.9738717
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3764715, 3.3879557
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1698780, 3.1779504
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9262600, 2.9147649
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5806189, 3.5993786

Time for backsubstitution: 14.51 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.453947067260742
rel_dist={6: [-2.09676497388166, 2.096767128916582]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 803

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6519131, upper bound: 1.6461280
time: 5.51 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6461293, upper bound: 1.6519130
time: 9.41 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.94 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.94
Output dim: 6, lower bound: -1.6519131, upper bound: 1.6461280
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.94
Output dim: 6, lower bound: -1.6461293, upper bound: 1.6519130

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2236366, 3.2256804
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8507576, 3.8519917
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.9018350, 2.8999586
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6289301, 2.6291041
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0695534, 3.0694270
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7695265, 2.7709961
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2666092, 3.2649813
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.8085327, 2.8074865
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6839185, 2.6865444
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2948704, 3.2949200

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6499967, upper bound: 1.6461239
time: 9.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6519090, upper bound: 1.6442591
time: 9.76 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2256804, 3.2236366
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8519917, 3.8507576
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8999591, 2.9018345
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6291046, 2.6289301
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0694275, 3.0695529
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7709970, 2.7695255
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2649813, 3.2666092
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.8074865, 2.8085327
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6865449, 2.6839180
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2949200, 3.2948713

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 819

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4554

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6440902, upper bound: 1.6518856
time: 6.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6460976, upper bound: 1.6498838
time: 9.42 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 30.45 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.45
Output dim: 6, lower bound: -1.6499967, upper bound: 1.6461239
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.45
Output dim: 6, lower bound: -1.6519090, upper bound: 1.6442591
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.45
Output dim: 6, lower bound: -1.6440902, upper bound: 1.6518856
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.45
Output dim: 6, lower bound: -1.6460976, upper bound: 1.6498838

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2007332, 3.2065945
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8805685, 3.8898449
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8668690, 2.8580093
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6419315, 2.6456070
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0250483, 3.0160251
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7847338, 2.7829714
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2636509, 3.2625160
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7902761, 2.7855930
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6498704, 2.6581674
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2705364, 3.2657204

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 110

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4554

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6479662, upper bound: 1.6460943
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6499672, upper bound: 1.6440876
time: 5.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2045517, 3.2027769
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8886099, 3.8818026
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8598852, 2.8649936
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6454334, 2.6421051
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0161514, 3.0249224
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7815008, 2.7862043
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2641439, 3.2620230
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7866387, 2.7892303
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6555409, 2.6524963
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2656717, 3.2705851

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 5717

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6518969, upper bound: 1.6339962
time: 7.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6416480, upper bound: 1.6442448
time: 6.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2258787, 3.2234478
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8519468, 3.8508053
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8998637, 2.9019332
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6293440, 2.6287003
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0698471, 3.0691500
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7714987, 2.7690392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2646704, 3.2669315
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.8064175, 2.8096523
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6869984, 2.6834812
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2943935, 3.2954540

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 4645

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6409032, upper bound: 1.6429001
time: 8.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6408879, upper bound: 1.6518783
time: 8.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2254906, 3.2236366
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8519917, 3.8507118
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8999591, 2.9017391
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6288738, 2.6289301
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0690241, 3.0695529
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7705097, 2.7695255
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2649813, 3.2662983
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.8074865, 2.8074636
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6861076, 2.6839180
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2949200, 3.2943439

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 819

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 805

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6460885, upper bound: 1.6308281
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6270645, upper bound: 1.6498754
time: 6.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 27.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.15
Output dim: 6, lower bound: -1.6479662, upper bound: 1.6460943
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.15
Output dim: 6, lower bound: -1.6499672, upper bound: 1.6440876
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.15
Output dim: 6, lower bound: -1.6518969, upper bound: 1.6339962
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.15
Output dim: 6, lower bound: -1.6416480, upper bound: 1.6442448
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.15
Output dim: 6, lower bound: -1.6409032, upper bound: 1.6429001
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.15
Output dim: 6, lower bound: -1.6408879, upper bound: 1.6518783
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 27.15
Output dim: 6, lower bound: -1.6460885, upper bound: 1.6308281
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 27.15
Output dim: 6, lower bound: -1.6270645, upper bound: 1.6498754

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2009315, 3.2064047
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8805218, 3.8898926
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8667746, 2.8581085
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6421700, 2.6453767
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0254679, 3.0156212
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7852373, 2.7824869
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2633381, 3.2628374
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7892070, 2.7867131
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6503253, 2.6577306
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2700090, 3.2663021

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 482

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6451747, upper bound: 1.6460935
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6479633, upper bound: 1.6433026
time: 15.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2005434, 3.2065945
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8805685, 3.8897991
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8668690, 2.8579149
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6417007, 2.6456070
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0246439, 3.0160251
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7842493, 2.7829714
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2636509, 3.2622032
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7902761, 2.7845240
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6494327, 2.6581674
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2705364, 3.2651930

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 110

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6473877, upper bound: 1.6439512
time: 10.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6498300, upper bound: 1.6415064
time: 11.52 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1957216, 3.1923704
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8864145, 3.8799181
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8598895, 2.8649988
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6453009, 2.6419506
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0124760, 3.0217710
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7805595, 2.7850728
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2740097, 3.2736320
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7820897, 2.7854409
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6649776, 2.6605110
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2582703, 3.2644196

Time for backsubstitution: 14.61 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 5717

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4645

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6518880, upper bound: 1.6330845
time: 13.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6509721, upper bound: 1.6339869
time: 10.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1941452, 3.1939468
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8867264, 3.8796062
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8598905, 2.8649983
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6452789, 2.6419725
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0129995, 3.0212474
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7803688, 2.7852616
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2757530, 3.2718897
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7828498, 2.7846813
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6635566, 2.6619334
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2595062, 3.2631836

Time for backsubstitution: 14.80 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 6185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4554

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6396180, upper bound: 1.6442145
time: 8.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6416186, upper bound: 1.6421961
time: 11.49 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2064543, 3.2097054
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8386631, 3.8419504
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8783336, 2.8745422
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6297126, 2.6291842
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0667200, 3.0662889
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7694817, 2.7712178
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2469854, 3.2457085
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7999887, 2.7994728
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6627703, 2.6660547
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2991962, 3.2994165

Time for backsubstitution: 14.78 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6396913, upper bound: 1.6429001
time: 8.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6409032, upper bound: 1.6416978
time: 8.33 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2121372, 3.2040234
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8430996, 3.8375225
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8724723, 2.8804040
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6298280, 2.6292939
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0672379, 3.0660224
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7736778, 2.7670217
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2434473, 3.2492476
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7962370, 2.8032331
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6695719, 2.6592541
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2983561, 3.3003540

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6396760, upper bound: 1.6518781
time: 7.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6408879, upper bound: 1.6506873
time: 43.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2244368, 3.2227612
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8469028, 3.8446074
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8999610, 2.9022226
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6208510, 2.6222420
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0698748, 3.0695543
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7713099, 2.7702045
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2598457, 3.2601376
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7919660, 2.7945247
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6882792, 2.6857610
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2839613, 3.2852077

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 6199

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6460763, upper bound: 1.6205016
time: 8.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6358277, upper bound: 1.6308155
time: 6.57 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2246122, 3.2225857
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8458872, 3.8456230
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.9004426, 2.9017415
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6221862, 2.6209073
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0690250, 3.0704036
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7711887, 2.7703257
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2588215, 3.2611628
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7945466, 2.7919436
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6879492, 2.6860909
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2857828, 3.2833862

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6244857, upper bound: 1.6497378
time: 9.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6269280, upper bound: 1.6472955
time: 9.83 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 34.14 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6451747, upper bound: 1.6460935
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6479633, upper bound: 1.6433026
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6473877, upper bound: 1.6439512
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6498300, upper bound: 1.6415064
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6518880, upper bound: 1.6330845
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6509721, upper bound: 1.6339869
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6396180, upper bound: 1.6442145
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6416186, upper bound: 1.6421961
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6396913, upper bound: 1.6429001
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6409032, upper bound: 1.6416978
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6396760, upper bound: 1.6518781
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6408879, upper bound: 1.6506873
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6460763, upper bound: 1.6205016
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6358277, upper bound: 1.6308155
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6244857, upper bound: 1.6497378
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 34.14
Output dim: 6, lower bound: -1.6269280, upper bound: 1.6472955

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1851530, 3.1874065
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8556786, 3.8600712
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8655610, 2.8589492
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6422195, 2.6463141
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0244808, 3.0141125
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7852383, 2.7825270
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2538576, 3.2554564
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7859678, 2.7828255
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6535583, 2.6602650
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2533703, 3.2442350

Time for backsubstitution: 14.77 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 4645

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6451689, upper bound: 1.6428911
time: 6.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6362291, upper bound: 1.6429099
time: 5.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1819324, 3.1906261
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8507004, 3.8650484
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8676152, 2.8568950
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6431084, 2.6454263
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0239592, 3.0146341
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7852774, 2.7824879
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2559576, 3.2533565
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7853203, 2.7834735
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6528583, 2.6609650
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2479429, 3.2496624

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 5717

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6479591, upper bound: 1.6433048
time: 7.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6479591, upper bound: 1.6414358
time: 10.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1981516, 3.2030067
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8802137, 3.8892679
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8680229, 2.8577900
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6410389, 2.6451688
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0244579, 3.0157442
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7834005, 2.7824011
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2600641, 3.2598238
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7882562, 2.7815332
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6498232, 2.6581569
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2690029, 3.2628689

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4645

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6473788, upper bound: 1.6430263
time: 9.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6464836, upper bound: 1.6439403
time: 14.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1969547, 3.2042027
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8800364, 3.8894453
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8667450, 2.8590679
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6412621, 2.6449461
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0243654, 3.0158372
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7836781, 2.7821226
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2612696, 3.2586184
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7872863, 2.7825031
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6494226, 2.6585574
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2682133, 3.2636585

Time for backsubstitution: 14.81 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 4645

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 110

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6498300, upper bound: 1.6374846
time: 10.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6458532, upper bound: 1.6415069
time: 16.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1946707, 3.1930695
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8870583, 3.8789454
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8604765, 2.8641214
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6465230, 2.6400967
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0126877, 3.0214419
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7816553, 2.7834125
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2744694, 3.2729321
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7820153, 2.7854877
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6642051, 2.6610246
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2568188, 3.2653742

Time for backsubstitution: 14.84 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6518856, upper bound: 1.6310667
time: 10.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6498680, upper bound: 1.6330843
time: 10.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1957216, 3.1913195
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8854399, 3.8799181
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8590117, 2.8649988
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6434474, 2.6419506
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0121469, 3.0217710
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7788992, 2.7850728
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2733097, 3.2736320
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7820897, 2.7853665
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6649776, 2.6597381
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2582703, 3.2629681

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 6185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 482

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6481805, upper bound: 1.6339847
time: 19.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6509692, upper bound: 1.6311960
time: 8.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1943436, 3.1937580
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8866816, 3.8796549
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8597960, 2.8650980
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6455173, 2.6417413
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0134192, 3.0208430
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7808733, 2.7847762
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2754412, 3.2722120
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7817807, 2.7858014
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6640096, 2.6614966
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2589798, 3.2637663

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 482

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6396111, upper bound: 1.6428978
time: 13.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6383017, upper bound: 1.6442078
time: 8.65 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 37.17 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6451689, upper bound: 1.6428911
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6362291, upper bound: 1.6429099
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6479591, upper bound: 1.6433048
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6479591, upper bound: 1.6414358
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6473788, upper bound: 1.6430263
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6464836, upper bound: 1.6439403
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6498300, upper bound: 1.6374846
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6458532, upper bound: 1.6415069
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6518856, upper bound: 1.6310667
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6498680, upper bound: 1.6330843
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6481805, upper bound: 1.6339847
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6509692, upper bound: 1.6311960
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6396111, upper bound: 1.6428978
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 37.17
Output dim: 6, lower bound: -1.6383017, upper bound: 1.6442078
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.17
Output dim: 6, lower bound: -1.6416186, upper bound: 1.6421961
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.17
Output dim: 6, lower bound: -1.6396913, upper bound: 1.6429001
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.17
Output dim: 6, lower bound: -1.6409032, upper bound: 1.6416978
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.17
Output dim: 6, lower bound: -1.6396760, upper bound: 1.6518781
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.17
Output dim: 6, lower bound: -1.6408879, upper bound: 1.6506873
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.17
Output dim: 6, lower bound: -1.6460763, upper bound: 1.6205016
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.17
Output dim: 6, lower bound: -1.6358277, upper bound: 1.6308155
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.17
Output dim: 6, lower bound: -1.6244857, upper bound: 1.6497378
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.17
Output dim: 6, lower bound: -1.6269280, upper bound: 1.6472955
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.269695281982422
rel_dist={6: [-1.652505574709414, 1.6525054132086048]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5847

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4930100, upper bound: 1.4916641
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4916648, upper bound: 1.4930094
time: 6.57 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.50 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.50
Output dim: 6, lower bound: -1.4930100, upper bound: 1.4916641
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.50
Output dim: 6, lower bound: -1.4916648, upper bound: 1.4930094

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1540089, 3.1545153
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7207184, 3.7193260
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8122587, 2.8118391
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5194311, 2.5143776
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9664955, 2.9671903
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6960020, 2.6949310
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1753931, 3.1756744
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6861105, 2.6855030
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6057816, 2.6029310
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1860914, 3.1852131

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 6185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6199

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4930080, upper bound: 1.4876071
time: 7.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4889490, upper bound: 1.4916626
time: 5.61 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1543245, 3.1540089
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7193260, 3.7202148
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8118391, 2.8120785
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5143776, 2.5176001
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9669409, 2.9664955
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6949310, 2.6956158
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1755638, 3.1753931
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6855030, 2.6858792
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6029320, 2.6047487
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1852131, 3.1857748

Time for backsubstitution: 14.89 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 803

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6199

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4916628, upper bound: 1.4889485
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4876076, upper bound: 1.4930079
time: 6.14 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 27.46 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.46
Output dim: 6, lower bound: -1.4930080, upper bound: 1.4876071
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.46
Output dim: 6, lower bound: -1.4889490, upper bound: 1.4916626
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 27.46
Output dim: 6, lower bound: -1.4916628, upper bound: 1.4889485
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 27.46
Output dim: 6, lower bound: -1.4876076, upper bound: 1.4930079

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1425028, 3.1450405
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7233629, 3.7225428
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8147345, 2.8173804
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4976912, 2.4885592
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9736023, 2.9767146
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6713848, 2.6667962
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1582088, 3.1560421
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6861992, 2.6856093
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6012769, 2.5989890
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1401224, 3.1449785

Time for backsubstitution: 14.88 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4554

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4912923, upper bound: 1.4875880
time: 8.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4929886, upper bound: 1.4858886
time: 8.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1445341, 3.1430101
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7239351, 3.7219715
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8178005, 2.8143153
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4936132, 2.4926372
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9760199, 2.9742970
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6678677, 2.6703134
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1557608, 3.1584902
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6862164, 2.6855912
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6018395, 2.5984273
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1458559, 3.1392441

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 805

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4889430, upper bound: 1.4776366
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4749178, upper bound: 1.4916568
time: 16.39 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1428185, 3.1445341
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7219715, 3.7234325
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8143158, 2.8176208
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4926376, 2.4917827
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9740467, 2.9760199
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6703129, 2.6674809
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1583796, 3.1557598
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6855907, 2.6859846
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5984273, 2.6008072
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1392450, 3.1455383

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 4645

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4916550, upper bound: 1.4879733
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4906884, upper bound: 1.4889422
time: 7.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1448488, 3.1425028
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7225428, 3.7228613
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8173809, 2.8145552
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4885597, 2.4958606
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9764643, 2.9736023
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6667957, 2.6709981
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1559315, 3.1582088
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6856098, 2.6859670
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5989881, 2.6002450
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1449785, 3.1398048

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4863946, upper bound: 1.4929971
time: 12.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4875953, upper bound: 1.4917428
time: 10.71 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 38.43 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.43
Output dim: 6, lower bound: -1.4912923, upper bound: 1.4875880
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.43
Output dim: 6, lower bound: -1.4929886, upper bound: 1.4858886
RS_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 38.43
Output dim: 6, lower bound: -1.4889430, upper bound: 1.4776366
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.43
Output dim: 6, lower bound: -1.4749178, upper bound: 1.4916568
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.43
Output dim: 6, lower bound: -1.4916550, upper bound: 1.4879733
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 38.43
Output dim: 6, lower bound: -1.4906884, upper bound: 1.4889422
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 38.43
Output dim: 6, lower bound: -1.4863946, upper bound: 1.4929971
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 38.43
Output dim: 6, lower bound: -1.4875953, upper bound: 1.4917428

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1426048, 3.1448517
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7233191, 3.7225685
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8146400, 2.8174310
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4978137, 2.4883299
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9738169, 2.9763117
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6716413, 2.6663117
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1578979, 3.1562061
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6851301, 2.6861825
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6015096, 2.5985527
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1395969, 3.1452847

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 819

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 110

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4912924, upper bound: 1.4845759
time: 8.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4882476, upper bound: 1.4875880
time: 6.56 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1423140, 3.1450405
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7233629, 3.7224979
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8147345, 2.8172855
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4974618, 2.4885592
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9731989, 2.9767146
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6708994, 2.6667962
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1582088, 3.1557312
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6861992, 2.6845407
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6008401, 2.5989890
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1401224, 3.1444530

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 482

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4909061, upper bound: 1.4858861
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4929865, upper bound: 1.4838067
time: 5.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1436129, 3.1419573
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7178307, 3.7166290
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8181629, 2.8143163
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4865904, 2.4846139
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9760199, 2.9749341
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6685457, 2.6710835
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1496010, 3.1530991
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6726313, 2.6700702
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6036820, 2.6005168
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1362648, 3.1282873

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 515

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4729491, upper bound: 1.4915203
time: 30.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4747770, upper bound: 1.4896936
time: 21.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1417675, 3.1447935
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7222118, 3.7224598
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8145370, 2.8167424
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4930863, 2.4899287
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9741249, 2.9756927
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6707201, 2.6658196
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1585503, 3.1550617
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6855183, 2.6860027
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5976548, 2.6009984
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1377916, 3.1458950

Time for backsubstitution: 14.51 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 803

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4910285, upper bound: 1.4830904
time: 12.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4867674, upper bound: 1.4873508
time: 14.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1219473, 3.1224642
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7523527, 3.7587023
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7806687, 2.7726054
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5015602, 2.5114861
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9297361, 2.9201994
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6811962, 2.6829729
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1529751, 3.1556206
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6664438, 2.6640744
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5649414, 2.5704503
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1194286, 3.1106071

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 6185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4863926, upper bound: 1.4914758
time: 10.53 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4848810, upper bound: 1.4929949
time: 10.92 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1248102, 3.1196012
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7583847, 3.7526712
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7754302, 2.7778440
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5041866, 2.5088596
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9230633, 2.9268723
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6787720, 2.6853971
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1533442, 3.1552515
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6637154, 2.6668024
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5691938, 2.5661974
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1157799, 3.1142559

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 110

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 803

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4869670, upper bound: 1.4868589
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4827071, upper bound: 1.4911144
time: 9.40 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 29.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4912924, upper bound: 1.4845759
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4882476, upper bound: 1.4875880
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4909061, upper bound: 1.4858861
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4929865, upper bound: 1.4838067
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4729491, upper bound: 1.4915203
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4747770, upper bound: 1.4896936
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4910285, upper bound: 1.4830904
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4867674, upper bound: 1.4873508
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4863926, upper bound: 1.4914758
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4848810, upper bound: 1.4929949
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4869670, upper bound: 1.4868589
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 29.22
Output dim: 6, lower bound: -1.4827071, upper bound: 1.4911144

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1426039, 3.1448507
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7233191, 3.7225676
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8146391, 2.8174314
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4978137, 2.4883294
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9738169, 2.9763112
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6716366, 2.6663060
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1578979, 3.1562061
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6851292, 2.6861815
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6015077, 2.5985522
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1395988, 3.1452856

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 5814

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4912844, upper bound: 1.4775601
time: 8.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4842791, upper bound: 1.4845685
time: 6.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1233149, 3.1284580
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.6935415, 3.6964083
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8150625, 2.8160734
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4981766, 2.4886093
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9716892, 2.9755974
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6709290, 2.6667953
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1503029, 3.1462498
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6823120, 2.6811395
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6033740, 2.6020484
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1180544, 3.1264544

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 515

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917215, upper bound: 1.4837917
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4929754, upper bound: 1.4825941
time: 7.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1409225, 3.1383686
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7174301, 3.7160950
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8189969, 2.8141928
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4859304, 2.4841199
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9758101, 2.9746542
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6676989, 2.6704445
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1460152, 3.1504173
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6703691, 2.6670804
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6039715, 2.6005063
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1345339, 3.1259632

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 900

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 482

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4721506, upper bound: 1.4915184
time: 11.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4716106, upper bound: 1.4889360
time: 5.55 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1114874, 3.1074352
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7493896, 3.7566309
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7782626, 2.7709270
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4974918, 2.5056448
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9173965, 2.9116096
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6756201, 2.6790910
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1526842, 3.1552105
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6610389, 2.6563087
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5630131, 2.5676789
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.0923071, 3.0917358

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 805

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4863897, upper bound: 1.4914545
time: 14.42 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4863895, upper bound: 1.4902528
time: 10.56 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1069183, 3.1120052
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7502813, 3.7557392
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7789912, 2.7701998
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4957161, 2.5074215
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9211464, 2.9078603
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6773157, 2.6773973
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1525640, 3.1553307
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6586785, 2.6586690
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5621700, 2.5685220
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1005611, 3.0834846

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 819

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 803

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4842488, upper bound: 1.4881100
time: 11.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4799956, upper bound: 1.4923678
time: 6.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 32.55 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 6, lower bound: -1.4912844, upper bound: 1.4775601
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 32.55
Output dim: 6, lower bound: -1.4842791, upper bound: 1.4845685
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 6, lower bound: -1.4917215, upper bound: 1.4837917
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 6, lower bound: -1.4929754, upper bound: 1.4825941
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 6, lower bound: -1.4721506, upper bound: 1.4915184
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 32.55
Output dim: 6, lower bound: -1.4716106, upper bound: 1.4889360
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 6, lower bound: -1.4863897, upper bound: 1.4914545
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 32.55
Output dim: 6, lower bound: -1.4863895, upper bound: 1.4902528
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 32.55
Output dim: 6, lower bound: -1.4842488, upper bound: 1.4881100
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 32.55
Output dim: 6, lower bound: -1.4799956, upper bound: 1.4923678

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1333809, 3.1344452
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7212009, 3.7206841
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8146448, 2.8174372
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4976826, 2.4881821
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9702711, 2.9731579
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6706476, 2.6651754
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1677628, 3.1673784
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6805801, 2.6822019
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6105890, 2.6065664
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1321974, 3.1388111

Time for backsubstitution: 14.76 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4905794, upper bound: 1.4775259
time: 5.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4912581, upper bound: 1.4768682
time: 6.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1004133, 3.1084194
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7233515, 3.7322521
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7783508, 2.7741227
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5111761, 2.5042353
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9249601, 2.9221954
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6853294, 2.6787729
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1473446, 3.1436615
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6631460, 2.6592455
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5693259, 2.5722542
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.0925055, 3.0972595

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 4645

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917139, upper bound: 1.4767740
time: 6.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4847692, upper bound: 1.4837834
time: 6.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1032772, 3.1055555
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7293835, 3.7262201
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7731133, 2.7793612
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5138025, 2.5016084
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9182873, 2.9288683
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6829052, 2.6811972
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1477146, 3.1432924
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6604185, 2.6619740
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5735793, 2.5680008
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.0888567, 3.1009083

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 6185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917237, upper bound: 1.4825949
time: 6.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4929567, upper bound: 1.4825947
time: 5.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1243382, 3.1193695
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.6913433, 3.6862745
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8139725, 2.8107090
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4866242, 2.4854817
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9746943, 2.9731464
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6676979, 2.6704741
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1365337, 3.1425114
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6669679, 2.6631937
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6070304, 2.6030397
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1188440, 3.1062031

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 4554

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 110

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4721505, upper bound: 1.4884785
time: 11.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4691362, upper bound: 1.4915180
time: 33.08 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1085987, 3.1054296
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7493801, 3.7579174
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7739782, 2.7647533
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.4974842, 2.5069489
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9121075, 2.9039922
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.6760778, 2.6790891
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1512671, 3.1542273
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6601362, 2.6550088
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5595551, 2.5652800
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.0906534, 3.0893497

Time for backsubstitution: 14.57 seconds
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.1755638122558594
rel_dist={6: [-1.493016446904722, 1.493015957466044]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2417.40 seconds
