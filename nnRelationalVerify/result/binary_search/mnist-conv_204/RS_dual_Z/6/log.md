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
execution time: IAR + LP analysis = 15.24 + 32.75 = 47.99 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.01 seconds, max iter: 100)

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
Binary search time: 211.78 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3340.23 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0905437, upper bound: 2.0967591
time: 22.10 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967590, upper bound: 2.0905435
time: 5.99 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 28.28 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 28.28
Output dim: 6, lower bound: -2.0905437, upper bound: 2.0967591
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 28.28
Output dim: 6, lower bound: -2.0967590, upper bound: 2.0905435

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2865715, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1556096, 3.1433868
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9779377, 2.9840670
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3390570, 3.3234873
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1690102, 3.1626449
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9177499, 2.9276738
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6020365, 3.5935230

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4554

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0885668, upper bound: 2.0967113
time: 9.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0904958, upper bound: 2.0947892
time: 5.75 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2968750, 4.2865705
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1433873, 3.1556096
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9840670, 2.9779377
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3234873, 3.3390574
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1626453, 3.1690106
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9276738, 2.9177499
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5935230, 3.6020365

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4554

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0947893, upper bound: 2.0904954
time: 9.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0967112, upper bound: 2.0885690
time: 6.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 30.27 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.27
Output dim: 6, lower bound: -2.0885668, upper bound: 2.0967113
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.27
Output dim: 6, lower bound: -2.0904958, upper bound: 2.0947892
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 30.27
Output dim: 6, lower bound: -2.0947893, upper bound: 2.0904954
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 30.27
Output dim: 6, lower bound: -2.0967112, upper bound: 2.0885690

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2865238, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1555152, 3.1436324
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9785290, 2.9838362
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3400955, 3.3230839
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1679420, 3.1654067
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9188724, 2.9272375
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6015100, 3.5949364

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0885540, upper bound: 2.0775482
time: 15.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0693900, upper bound: 2.0966981
time: 20.30 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2865715, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1556096, 3.1432924
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9777069, 2.9840670
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3386545, 3.3234873
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1690102, 3.1615758
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9173121, 2.9276738
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.6020365, 3.5929966

Time for backsubstitution: 15.09 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0904830, upper bound: 2.0756263
time: 7.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0713195, upper bound: 2.0947781
time: 10.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2968750, 4.2866879
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1432929, 3.1558552
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9846592, 2.9777074
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3245258, 3.3386545
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1615753, 3.1717725
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9287963, 2.9173136
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5929966, 3.6034508

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0947759, upper bound: 2.0713197
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0756264, upper bound: 2.0904851
time: 6.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2968750, 4.2865238
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1433873, 3.1555152
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9838371, 2.9779377
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3230839, 3.3390574
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1626453, 3.1679416
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9272361, 2.9177499
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5935230, 3.6015100

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0966979, upper bound: 2.0693896
time: 7.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0775484, upper bound: 2.0885538
time: 10.35 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 32.58 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 6, lower bound: -2.0885540, upper bound: 2.0775482
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 6, lower bound: -2.0693900, upper bound: 2.0966981
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 6, lower bound: -2.0904830, upper bound: 2.0756263
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 6, lower bound: -2.0713195, upper bound: 2.0947781
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 6, lower bound: -2.0947759, upper bound: 2.0713197
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 6, lower bound: -2.0756264, upper bound: 2.0904851
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 6, lower bound: -2.0966979, upper bound: 2.0693896
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 32.58
Output dim: 6, lower bound: -2.0775484, upper bound: 2.0885538

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2840939, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1555181, 3.1436367
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9783964, 2.9836650
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3360281, 3.3199329
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1633925, 3.1621876
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9293761, 2.9352531
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5941105, 3.5896988

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0885498, upper bound: 2.0775479
time: 6.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0885504, upper bound: 2.0713192
time: 7.78 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2846394, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1555200, 3.1436357
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9783583, 2.9837031
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3369446, 3.3190165
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1647220, 3.1608582
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9268870, 2.9377418
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5962715, 3.5875359

Time for backsubstitution: 15.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0693858, upper bound: 2.0966957
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0693864, upper bound: 2.0904786
time: 5.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2841396, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1556134, 3.1432972
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9775743, 2.9838963
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3345871, 3.3203359
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1644616, 3.1583567
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9278178, 2.9356899
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5946369, 3.5877571

Time for backsubstitution: 14.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0904665, upper bound: 2.0756235
time: 7.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0904671, upper bound: 2.0693979
time: 7.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2846851, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1556144, 3.1432958
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9775362, 2.9839344
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3355026, 3.3194199
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1657910, 3.1570272
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9253287, 2.9381785
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5967979, 3.5855961

Time for backsubstitution: 14.75 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0713075, upper bound: 2.0947733
time: 8.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0713081, upper bound: 2.0885636
time: 6.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2968750, 4.2848043
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1432958, 3.1558599
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9845257, 2.9775362
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3204575, 3.3355026
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1570268, 3.1685529
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9393001, 2.9253292
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5855961, 3.5982113

Time for backsubstitution: 14.69 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0885614, upper bound: 2.0713078
time: 5.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0947732, upper bound: 2.0713074
time: 21.53 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2968750, 4.2842569
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1432977, 3.1558580
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9844875, 2.9775743
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3213739, 3.3345866
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1583562, 3.1672235
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9368110, 2.9278178
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5877571, 3.5960503

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0693974, upper bound: 2.0904673
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0756237, upper bound: 2.0904672
time: 5.11 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2968750, 4.2846403
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1433902, 3.1555200
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9837036, 2.9777670
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3190165, 3.3359060
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1580958, 3.1647220
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9377418, 2.9257660
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5861225, 3.5962715

Time for backsubstitution: 14.64 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0904784, upper bound: 2.0693862
time: 10.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0966952, upper bound: 2.0693861
time: 9.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2968750, 4.2840939
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1433921, 3.1555185
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9836655, 2.9778051
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3199329, 3.3349895
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1594253, 3.1633925
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9352527, 2.9282546
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5882835, 3.5941095

Time for backsubstitution: 14.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0713194, upper bound: 2.0885502
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0775457, upper bound: 2.0885502
time: 14.99 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 37.79 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0885498, upper bound: 2.0775479
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0885504, upper bound: 2.0713192
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0693858, upper bound: 2.0966957
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0693864, upper bound: 2.0904786
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0904665, upper bound: 2.0756235
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0904671, upper bound: 2.0693979
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0713075, upper bound: 2.0947733
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0713081, upper bound: 2.0885636
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0885614, upper bound: 2.0713078
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0947732, upper bound: 2.0713074
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0693974, upper bound: 2.0904673
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0756237, upper bound: 2.0904672
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0904784, upper bound: 2.0693862
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0966952, upper bound: 2.0693861
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0713194, upper bound: 2.0885502
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 37.79
Output dim: 6, lower bound: -2.0775457, upper bound: 2.0885502

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2840843, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1537523, 3.1374660
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9783888, 2.9867177
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3338451, 3.3123155
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1630211, 3.1608868
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9259191, 2.9342651
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5934315, 3.5873137

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0885388, upper bound: 2.0644932
time: 12.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0755869, upper bound: 2.0775352
time: 5.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2871094, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1493473, 3.1418710
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9814482, 2.9836574
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3284111, 3.3177500
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1620913, 3.1618037
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9283881, 2.9317951
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5917244, 3.5890198

Time for backsubstitution: 14.55 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0885393, upper bound: 2.0583735
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0755874, upper bound: 2.0713093
time: 5.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2846298, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1537542, 3.1374645
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9783506, 2.9867563
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3347616, 3.3113990
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1643505, 3.1595569
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9234309, 2.9367542
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5955935, 3.5851517

Time for backsubstitution: 14.53 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0693748, upper bound: 2.0836450
time: 8.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0564391, upper bound: 2.0966850
time: 7.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2876568, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1493492, 3.1418695
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9814100, 2.9836955
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3293266, 3.3168335
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1634207, 3.1604738
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9258990, 2.9342842
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5938873, 3.5868587

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0693753, upper bound: 2.0775211
time: 23.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0564396, upper bound: 2.0904680
time: 8.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2841282, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1538467, 3.1371260
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9775667, 2.9869490
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3324032, 3.3127203
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1640902, 3.1570559
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9243598, 2.9347019
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5939579, 3.5853729

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0904560, upper bound: 2.0625675
time: 6.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0775102, upper bound: 2.0756128
time: 6.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2871552, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1494417, 3.1415310
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9806261, 2.9838886
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3269691, 3.3181553
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1631603, 3.1579728
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9268279, 2.9322319
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5922518, 3.5870800

Time for backsubstitution: 14.59 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0904567, upper bound: 2.0564504
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0775107, upper bound: 2.0693884
time: 6.29 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2846756, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1538486, 3.1371250
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9775286, 2.9869871
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3333197, 3.3118043
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1654196, 3.1557260
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9218717, 2.9371910
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5961199, 3.5832109

Time for backsubstitution: 14.52 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0712971, upper bound: 2.0817194
time: 9.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0583624, upper bound: 2.0947617
time: 12.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2877026, 4.2968750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1494436, 3.1415300
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9805880, 2.9839268
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3278856, 3.3172388
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1644897, 3.1566434
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9243398, 2.9347210
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5944138, 3.5849171

Time for backsubstitution: 14.65 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0712977, upper bound: 2.0755980
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -2.0583629, upper bound: 2.0885507
time: 6.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2435699, 3.2435699
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -4.2968750, 4.2878199
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -3.1415300, 3.1496887
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.9845181, 2.9805875
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.3182755, 3.3278856
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.9095392, 2.9095392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.4539471, 3.4539471
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -3.1566429, 3.1672521
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.9358430, 2.9243398
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.5849171, 3.5958271

Time for backsubstitution: 14.78 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=3.453947067260742
rel_dist={6: [-2.09676497388166, 2.096767128916582]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6506532, upper bound: 1.6525026
time: 16.16 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6525029, upper bound: 1.6506533
time: 6.76 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 23.10 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 23.10
Output dim: 6, lower bound: -1.6506532, upper bound: 1.6525026
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 23.10
Output dim: 6, lower bound: -1.6525029, upper bound: 1.6506533

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2066555, 3.2104731
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8841619, 3.8922043
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8704276, 2.8634429
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6424351, 2.6459370
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0249224, 3.0160255
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7889881, 2.7857561
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2667389, 3.2672319
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7922878, 2.7886515
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6574626, 2.6631336
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2706776, 3.2658119

Time for backsubstitution: 14.67 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 4554

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6486149, upper bound: 1.6524733
time: 10.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6506238, upper bound: 1.6504803
time: 6.09 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2104731, 3.2066545
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8922043, 3.8841619
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8634429, 2.8704271
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6459370, 2.6424346
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0160255, 3.0249224
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7857561, 2.7889881
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2672319, 3.2667389
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7886505, 2.7922888
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6631341, 2.6574631
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2658119, 3.2706776

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 4554

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6504803, upper bound: 1.6506236
time: 9.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6524738, upper bound: 1.6486148
time: 8.83 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 33.15 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 33.15
Output dim: 6, lower bound: -1.6486149, upper bound: 1.6524733
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 33.15
Output dim: 6, lower bound: -1.6506238, upper bound: 1.6504803
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 33.15
Output dim: 6, lower bound: -1.6504803, upper bound: 1.6506236
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 33.15
Output dim: 6, lower bound: -1.6524738, upper bound: 1.6486148

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2068529, 3.2102833
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8841152, 3.8922520
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8703332, 2.8635426
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6426735, 2.6457062
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0253429, 3.0156221
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7894917, 2.7852716
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2664261, 3.2675524
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7912197, 2.7897716
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6579165, 2.6626973
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2701511, 3.2663946

Time for backsubstitution: 14.54 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6486024, upper bound: 1.6422126
time: 8.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6383538, upper bound: 1.6524635
time: 6.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2064648, 3.2104731
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8841619, 3.8921585
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8704276, 2.8633485
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6422043, 2.6459370
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0245190, 3.0160255
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7885036, 2.7857561
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2667389, 3.2669191
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7922878, 2.7875829
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6570258, 2.6631336
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2706776, 3.2652855

Time for backsubstitution: 14.68 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6506113, upper bound: 1.6402191
time: 12.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6403628, upper bound: 1.6504677
time: 9.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2106705, 3.2064648
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8921585, 3.8842087
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8633485, 2.8705273
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6461754, 2.6422038
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0164461, 3.0245194
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7862597, 2.7885036
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2669191, 3.2670603
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7875824, 2.7934089
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6635871, 2.6570263
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2652855, 3.2712593

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6504681, upper bound: 1.6403629
time: 9.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6402193, upper bound: 1.6506112
time: 6.31 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2102823, 3.2066545
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8922043, 3.8841152
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8634429, 2.8703327
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6457062, 2.6424346
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0156221, 3.0249224
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7852716, 2.7889881
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2672319, 3.2664261
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7886505, 2.7912202
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6626964, 2.6574631
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2658119, 3.2701511

Time for backsubstitution: 14.83 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6524617, upper bound: 1.6383538
time: 11.56 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6422128, upper bound: 1.6486023
time: 6.46 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 33.06 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 6, lower bound: -1.6486024, upper bound: 1.6422126
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 6, lower bound: -1.6383538, upper bound: 1.6524635
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 6, lower bound: -1.6506113, upper bound: 1.6402191
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 6, lower bound: -1.6403628, upper bound: 1.6504677
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 6, lower bound: -1.6504681, upper bound: 1.6403629
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 6, lower bound: -1.6402193, upper bound: 1.6506112
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 6, lower bound: -1.6524617, upper bound: 1.6383538
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 33.06
Output dim: 6, lower bound: -1.6422128, upper bound: 1.6486023

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1980267, 3.1998796
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8819199, 3.8903685
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8703361, 2.8635468
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6425409, 2.6455517
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0216675, 3.0124707
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7885485, 2.7841392
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2762928, 3.2791624
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7866712, 2.7859821
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6673541, 2.6707129
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2627506, 3.2602291

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6485998, upper bound: 1.6422128
time: 7.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6485997, upper bound: 1.6403627
time: 5.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1964502, 3.2014561
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8822327, 3.8900557
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8703370, 2.8635459
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6425190, 2.6455736
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0221920, 3.0119472
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7883596, 2.7843289
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2780352, 3.2774191
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7874303, 2.7852225
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6659331, 2.6721349
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2639866, 3.2589951

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6383514, upper bound: 1.6524635
time: 5.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6383513, upper bound: 1.6506113
time: 6.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1976385, 3.2000694
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8819647, 3.8902750
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8704305, 2.8633528
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6420708, 2.6457825
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0208445, 3.0128736
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7875605, 2.7846246
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2766047, 3.2785282
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7877393, 2.7837934
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6664634, 2.6711493
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2632771, 3.2591209

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6506077, upper bound: 1.6402212
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6506076, upper bound: 1.6383549
time: 7.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1960621, 3.2016459
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8822775, 3.8899622
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8704314, 2.8633518
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6420498, 2.6458044
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0213680, 3.0123501
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7873716, 2.7848144
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2783470, 3.2767849
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7884994, 2.7830334
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6650424, 2.6725717
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2645130, 3.2578850

Time for backsubstitution: 14.82 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6403592, upper bound: 1.6504681
time: 6.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6403591, upper bound: 1.6486036
time: 6.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2018442, 3.1960621
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8899632, 3.8823261
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8633523, 2.8705311
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6460438, 2.6420488
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0127707, 3.0213680
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7853174, 2.7873712
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2767849, 3.2786694
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7830338, 2.7896194
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6730266, 2.6650419
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2578850, 3.2650948

Time for backsubstitution: 14.85 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6486034, upper bound: 1.6403591
time: 10.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6504681, upper bound: 1.6403611
time: 6.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2002678, 3.1976385
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8902750, 3.8820133
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8633523, 2.8705301
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6460209, 2.6420708
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0132942, 3.0208445
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7851267, 2.7875609
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2785282, 3.2769260
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7837930, 2.7888598
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6716037, 2.6664643
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2591209, 3.2638588

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6383549, upper bound: 1.6506078
time: 6.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6402193, upper bound: 1.6506075
time: 8.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.2014561, 3.1962509
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8900080, 3.8822327
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8634467, 2.8703370
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6455736, 2.6422806
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0119467, 3.0217710
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7843294, 2.7878566
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2770967, 3.2780352
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7841020, 2.7874308
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6721339, 2.6654787
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2584114, 3.2639856

Time for backsubstitution: 14.94 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6506113, upper bound: 1.6383510
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6524617, upper bound: 1.6383534
time: 7.61 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1998796, 3.1978273
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8903198, 3.8819199
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8634477, 2.8703365
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6455517, 2.6423025
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0124702, 3.0212474
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7841387, 2.7880464
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2788401, 3.2762928
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7848620, 2.7866707
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6707129, 2.6669006
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2596474, 3.2627506

Time for backsubstitution: 14.92 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6403628, upper bound: 1.6485994
time: 7.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6422128, upper bound: 1.6485999
time: 9.27 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 31.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6485998, upper bound: 1.6422128
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6485997, upper bound: 1.6403627
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6383514, upper bound: 1.6524635
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6383513, upper bound: 1.6506113
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6506077, upper bound: 1.6402212
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6506076, upper bound: 1.6383549
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6403592, upper bound: 1.6504681
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6403591, upper bound: 1.6486036
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6486034, upper bound: 1.6403591
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6504681, upper bound: 1.6403611
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6383549, upper bound: 1.6506078
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6402193, upper bound: 1.6506075
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6506113, upper bound: 1.6383510
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6524617, upper bound: 1.6383534
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6403628, upper bound: 1.6485994
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 31.86
Output dim: 6, lower bound: -1.6422128, upper bound: 1.6485999

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1951380, 3.1981678
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8819103, 3.8920870
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8666830, 2.8573756
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6425333, 2.6472926
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0171556, 3.0048532
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7891579, 2.7841363
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2748766, 3.2783222
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7859011, 2.7846813
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6638980, 2.6686668
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2613411, 3.2578449

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6485935, upper bound: 1.6345225
time: 13.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6409053, upper bound: 1.6422066
time: 8.33 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1963148, 3.1969919
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8836393, 3.8903580
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8641653, 2.8598928
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6442814, 2.6455441
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0140505, 3.0079589
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7885456, 2.7847486
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2754536, 3.2777452
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7853699, 2.7852054
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6653075, 2.6672554
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2603655, 3.2588196

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6485934, upper bound: 1.6326922
time: 12.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6409052, upper bound: 1.6403591
time: 7.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1935616, 3.1997442
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8822222, 3.8917742
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8666830, 2.8573751
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6425114, 2.6473145
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0176792, 3.0043297
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7889690, 2.7843251
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2766190, 3.2765799
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7866602, 2.7839217
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6624751, 2.6700888
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2625761, 3.2566099

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6383451, upper bound: 1.6447713
time: 7.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6306556, upper bound: 1.6524577
time: 6.35 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1947384, 3.1985683
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8839512, 3.8900461
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8641663, 2.8598919
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6442595, 2.6455660
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0145741, 3.0074353
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7883568, 2.7849374
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2771969, 3.2760029
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7861300, 2.7844458
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6638856, 2.6686773
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2616014, 3.2575846

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6383450, upper bound: 1.6429406
time: 9.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6306555, upper bound: 1.6506057
time: 9.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1947498, 3.1983566
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8819551, 3.8919935
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8667765, 2.8571815
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6420641, 2.6475239
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0163326, 3.0052586
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7881699, 2.7846227
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2751894, 3.2776890
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7869701, 2.7824922
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6630063, 2.6691031
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2618675, 3.2567368

Time for backsubstitution: 14.86 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6506020, upper bound: 1.6325251
time: 9.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6429378, upper bound: 1.6402148
time: 7.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1959267, 3.1971807
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8836842, 3.8902645
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8642597, 2.8596988
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6438112, 2.6457748
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0132275, 3.0083642
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7875576, 2.7852349
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2757664, 3.2771120
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7864389, 2.7830162
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6644168, 2.6676917
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2608929, 3.2577114

Time for backsubstitution: 14.90 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6506019, upper bound: 1.6306584
time: 7.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6429377, upper bound: 1.6383486
time: 7.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1931734, 3.1999331
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8822680, 3.8916807
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8667774, 2.8571806
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6420412, 2.6475458
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0168562, 3.0047350
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7879810, 2.7848125
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2769318, 3.2759457
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7877293, 2.7817326
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6615844, 2.6705256
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2631025, 3.2555008

Time for backsubstitution: 14.87 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6403535, upper bound: 1.6427744
time: 7.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6326886, upper bound: 1.6504614
time: 10.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1943502, 3.1987572
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8839970, 3.8899527
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8642607, 2.8596983
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6437902, 2.6457968
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0137510, 3.0078406
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7873688, 2.7854238
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2775087, 3.2753687
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7871981, 2.7822566
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6629949, 2.6691141
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2621279, 3.2564754

Time for backsubstitution: 14.91 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6403534, upper bound: 1.6409085
time: 10.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6326885, upper bound: 1.6485991
time: 6.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1989555, 3.1943502
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.8899517, 3.8840446
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.8596983, 2.8643603
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.6460352, 2.6437898
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -3.0082588, 3.0137506
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7859249, 2.7873683
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.2753687, 3.2778301
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.7822561, 2.7883186
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.6695685, 2.6629949
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.2564754, 3.2627106

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6485970, upper bound: 1.6326879
time: 20.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.6409086, upper bound: 1.6403533
time: 11.09 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 46.93 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6485935, upper bound: 1.6345225
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6409053, upper bound: 1.6422066
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6485934, upper bound: 1.6326922
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6409052, upper bound: 1.6403591
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6383451, upper bound: 1.6447713
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6306556, upper bound: 1.6524577
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6383450, upper bound: 1.6429406
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6306555, upper bound: 1.6506057
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6506020, upper bound: 1.6325251
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6429378, upper bound: 1.6402148
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6506019, upper bound: 1.6306584
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6429377, upper bound: 1.6383486
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6403535, upper bound: 1.6427744
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6326886, upper bound: 1.6504614
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6403534, upper bound: 1.6409085
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6326885, upper bound: 1.6485991
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6485970, upper bound: 1.6326879
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 46.93
Output dim: 6, lower bound: -1.6409086, upper bound: 1.6403533
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.93
Output dim: 6, lower bound: -1.6504681, upper bound: 1.6403611
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 46.93
Output dim: 6, lower bound: -1.6383549, upper bound: 1.6506078
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.93
Output dim: 6, lower bound: -1.6402193, upper bound: 1.6506075
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 46.93
Output dim: 6, lower bound: -1.6506113, upper bound: 1.6383510
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.93
Output dim: 6, lower bound: -1.6524617, upper bound: 1.6383534
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 46.93
Output dim: 6, lower bound: -1.6403628, upper bound: 1.6485994
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 46.93
Output dim: 6, lower bound: -1.6422128, upper bound: 1.6485999
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=3.269695281982422
rel_dist={6: [-1.652505574709414, 1.6525054132086048]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5814
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 5814

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917514, upper bound: 1.4930053
time: 6.64 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4930054, upper bound: 1.4917514
time: 6.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 13.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 13.02
Output dim: 6, lower bound: -1.4917514, upper bound: 1.4930053
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 13.02
Output dim: 6, lower bound: -1.4930054, upper bound: 1.4917514

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1314201, 3.1342840
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7500267, 3.7560577
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7753673, 2.7701283
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5306005, 2.5332270
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9202108, 2.9135380
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7100153, 2.7075911
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1726074, 3.1729774
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6667156, 2.6639867
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5707002, 2.5749540
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1602240, 3.1565752

Time for backsubstitution: 14.95 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 4554

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4900285, upper bound: 1.4929862
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917322, upper bound: 1.4912881
time: 7.83 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1342840, 3.1314201
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7560577, 3.7500267
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7701287, 2.7753668
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5332270, 2.5306001
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9135380, 2.9202108
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7075911, 2.7100153
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1729774, 3.1726074
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6639862, 2.6667151
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5749536, 2.5707006
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1565752, 3.1602240

Time for backsubstitution: 14.97 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 4554
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 4554

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4900305, upper bound: 1.4917316
time: 6.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4929861, upper bound: 1.4900287
time: 7.31 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 29.30 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.30
Output dim: 6, lower bound: -1.4900285, upper bound: 1.4929862
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.30
Output dim: 6, lower bound: -1.4917322, upper bound: 1.4912881
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 29.30
Output dim: 6, lower bound: -1.4900305, upper bound: 1.4917316
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 29.30
Output dim: 6, lower bound: -1.4929861, upper bound: 1.4900287

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1315212, 3.1340933
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7499800, 3.7560816
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7752728, 2.7701797
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5307217, 2.5329962
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9204254, 2.9131346
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7102718, 2.7071066
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1722946, 3.1731396
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6656456, 2.6645598
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5709319, 2.5745173
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1596975, 3.1568804

Time for backsubstitution: 15.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4900197, upper bound: 1.4859825
time: 12.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4830702, upper bound: 1.4929784
time: 9.71 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1312304, 3.1342840
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7500267, 3.7560120
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7753673, 2.7700338
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5303688, 2.5332270
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9198074, 2.9135380
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7095308, 2.7075911
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1726074, 3.1726646
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6667156, 2.6629181
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5702643, 2.5749540
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1602240, 3.1560488

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.20 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917245, upper bound: 1.4842770
time: 7.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4847799, upper bound: 1.4912802
time: 9.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1343842, 3.1312304
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7560120, 3.7500496
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7700343, 2.7754178
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5333481, 2.5303693
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9137526, 2.9198074
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7078476, 2.7095308
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1726646, 3.1727705
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6629181, 2.6672878
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5751853, 2.5702643
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1560488, 3.1605291

Time for backsubstitution: 14.96 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4912804, upper bound: 1.4847796
time: 8.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4842751, upper bound: 1.4917264
time: 7.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1340933, 3.1314201
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7560577, 3.7499790
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7701287, 2.7752724
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5329971, 2.5306001
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9131346, 2.9202108
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7071066, 2.7100153
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1729774, 3.1722946
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6639862, 2.6656461
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5745177, 2.5707006
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1565752, 3.1596975

Time for backsubstitution: 15.05 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 819
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 819

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4929786, upper bound: 1.4830704
time: 8.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4859826, upper bound: 1.4900199
time: 10.78 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 34.97 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 34.97
Output dim: 6, lower bound: -1.4900197, upper bound: 1.4859825
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 34.97
Output dim: 6, lower bound: -1.4830702, upper bound: 1.4929784
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 34.97
Output dim: 6, lower bound: -1.4917245, upper bound: 1.4842770
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 34.97
Output dim: 6, lower bound: -1.4847799, upper bound: 1.4912802
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 34.97
Output dim: 6, lower bound: -1.4912804, upper bound: 1.4847796
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 34.97
Output dim: 6, lower bound: -1.4842751, upper bound: 1.4917264
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 34.97
Output dim: 6, lower bound: -1.4929786, upper bound: 1.4830704
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 34.97
Output dim: 6, lower bound: -1.4859826, upper bound: 1.4900199

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1211176, 3.1248732
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7480965, 3.7539644
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7752767, 2.7701826
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5305729, 2.5328636
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9172745, 2.9095907
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7091398, 2.7061162
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1834679, 3.1830063
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6616664, 2.6600108
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5789485, 2.5835996
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1532230, 3.1494808

Time for backsubstitution: 14.66 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4830702, upper bound: 1.4929595
time: 11.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4830701, upper bound: 1.4917265
time: 10.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1220093, 3.1238794
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7479067, 3.7541285
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7753701, 2.7700377
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5302372, 2.5330782
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9162636, 2.9103866
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7085409, 2.7064595
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1824722, 3.1838379
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6621661, 2.6589384
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5793452, 2.5829697
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1528244, 3.1495752

Time for backsubstitution: 14.93 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.21 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917245, upper bound: 1.4842493
time: 5.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917244, upper bound: 1.4830742
time: 18.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1208277, 3.1250620
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7481413, 3.7538939
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7753711, 2.7700372
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5302200, 2.5330944
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9166565, 2.9099941
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7083979, 2.7066016
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1837797, 3.1825304
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6627355, 2.6583691
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5782790, 2.5840359
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1537514, 3.1486492

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4847763, upper bound: 1.4912579
time: 47.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4847762, upper bound: 1.4900218
time: 19.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1251640, 3.1208267
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7538939, 3.7481661
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7700372, 2.7754216
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5332155, 2.5302200
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9102087, 2.9166565
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7068567, 2.7083983
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1825304, 3.1839437
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6583686, 2.6633086
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5842681, 2.5782800
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1486492, 3.1540556

Time for backsubstitution: 15.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4900218, upper bound: 1.4847763
time: 5.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4912579, upper bound: 1.4847763
time: 6.15 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1239815, 3.1220093
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7541285, 3.7479324
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7700381, 2.7754211
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5331993, 2.5302367
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9106007, 2.9162636
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7067156, 2.7085404
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1838379, 3.1826363
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6589389, 2.6627388
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5832000, 2.5793462
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1495743, 3.1531296

Time for backsubstitution: 14.72 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4830723, upper bound: 1.4917242
time: 5.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4842492, upper bound: 1.4917243
time: 11.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1248732, 3.1210165
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7539387, 3.7480965
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7701316, 2.7752762
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5328636, 2.5304513
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9095907, 2.9170594
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7061167, 2.7088838
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1828423, 3.1834679
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6594377, 2.6616669
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5835986, 2.5787163
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1491756, 3.1532240

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 900
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 900

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917268, upper bound: 1.4830701
time: 14.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4929595, upper bound: 1.4830706
time: 6.17 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 35.22 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4830702, upper bound: 1.4929595
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4830701, upper bound: 1.4917265
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4917245, upper bound: 1.4842493
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4917244, upper bound: 1.4830742
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4847763, upper bound: 1.4912579
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4847762, upper bound: 1.4900218
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4900218, upper bound: 1.4847763
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4912579, upper bound: 1.4847763
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4830723, upper bound: 1.4917242
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4842492, upper bound: 1.4917243
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4917268, upper bound: 1.4830701
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 35.22
Output dim: 6, lower bound: -1.4929595, upper bound: 1.4830706

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1182299, 3.1228666
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7480860, 3.7552500
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7709932, 2.7640119
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5305653, 2.5341673
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9119854, 2.9019732
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7095947, 2.7061133
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1820517, 3.1820221
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6607637, 2.6587095
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5754905, 2.5812006
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1515703, 3.1470957

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4830657, upper bound: 1.4872807
time: 7.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4773748, upper bound: 1.4929558
time: 6.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1191120, 3.1219845
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7493830, 3.7539539
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7691050, 2.7658997
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5318766, 2.5328560
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9096565, 2.9043026
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7091370, 2.7065721
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1824846, 3.1815891
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6603661, 2.6591029
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5765481, 2.5801420
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1508389, 3.1478271

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4830655, upper bound: 1.4860494
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4773746, upper bound: 1.4917230
time: 10.99 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1191216, 3.1218739
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7478981, 3.7554140
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7710867, 2.7638669
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5302286, 2.5343819
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9109755, 2.9027710
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7089958, 2.7064576
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1810570, 3.1828537
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6612635, 2.6576376
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5758891, 2.5805707
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1511707, 3.1471901

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917205, upper bound: 1.4785449
time: 8.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4860477, upper bound: 1.4842439
time: 6.49 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1200037, 3.1209917
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7491941, 3.7541180
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7691994, 2.7657547
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5315399, 2.5330706
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9086466, 2.9051003
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7085381, 2.7069163
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1814899, 3.1824217
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6608648, 2.6580310
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5769467, 2.5795121
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1504393, 3.1479216

Time for backsubstitution: 14.57 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917204, upper bound: 1.4773767
time: 7.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4860477, upper bound: 1.4830679
time: 6.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1210928, 3.1200037
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7541180, 3.7492189
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7657547, 2.7692499
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5331917, 2.5315399
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9053125, 2.9086461
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7071705, 2.7085376
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1824217, 3.1816530
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6580305, 2.6614380
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5797439, 2.5769463
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1479216, 3.1507444

Time for backsubstitution: 14.56 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4830677, upper bound: 1.4860478
time: 7.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4773768, upper bound: 1.4917223
time: 5.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1219749, 3.1191216
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7554150, 3.7479219
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7638674, 2.7711382
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5345030, 2.5302291
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9029837, 2.9109755
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7067127, 2.7089963
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1828547, 3.1812201
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6576376, 2.6618361
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5808024, 2.5758886
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1471901, 3.1514759

Time for backsubstitution: 14.58 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4842446, upper bound: 1.4860479
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4785430, upper bound: 1.4917203
time: 8.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1219845, 3.1190109
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7539291, 3.7493830
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7658491, 2.7691050
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5328560, 2.5317545
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9043026, 2.9094439
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7065716, 2.7088819
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1814270, 3.1824846
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6585302, 2.6603661
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5801415, 2.5763168
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1475220, 3.1508389

Time for backsubstitution: 14.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4917228, upper bound: 1.4773751
time: 6.37 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4860500, upper bound: 1.4830656
time: 5.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.1228666, 3.1181278
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7552261, 3.7480860
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7639608, 2.7709932
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5341673, 2.5304437
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9019737, 2.9117732
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7061138, 2.7093406
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1818600, 3.1820517
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6581364, 2.6607642
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5812001, 2.5752587
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1467905, 3.1515703

Time for backsubstitution: 14.71 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 5717
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 5717

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4929555, upper bound: 1.4773746
time: 6.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4872802, upper bound: 1.4830655
time: 6.24 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 27.83 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4830657, upper bound: 1.4872807
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4773748, upper bound: 1.4929558
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4830655, upper bound: 1.4860494
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4773746, upper bound: 1.4917230
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4917205, upper bound: 1.4785449
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4860477, upper bound: 1.4842439
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4917204, upper bound: 1.4773767
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4860477, upper bound: 1.4830679
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4830677, upper bound: 1.4860478
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4773768, upper bound: 1.4917223
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4842446, upper bound: 1.4860479
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4785430, upper bound: 1.4917203
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4917228, upper bound: 1.4773751
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4860500, upper bound: 1.4830656
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4929555, upper bound: 1.4773746
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 27.83
Output dim: 6, lower bound: -1.4872802, upper bound: 1.4830655

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.0971556, 3.0975313
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7345390, 3.7383795
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7406774, 2.7380919
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5310493, 2.5347338
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9092484, 2.8988476
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7064800, 2.6998510
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1608286, 3.1634531
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6475301, 2.6482897
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5487862, 2.5493951
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1555347, 3.1517630

Time for backsubstitution: 15.04 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.22 seconds

### Candidate
type: RSZ, layer: 1, pos: 805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4773690, upper bound: 1.4789225
time: 9.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4632758, upper bound: 1.4929515
time: 7.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -7.2269325, -3.9833627, -7.2269325, -3.9833627, -3.0980387, 3.0966492
1: -13.8976402, -9.6007652, -13.8976402, -9.6007652, -3.7358351, 3.7370834
2: -7.1633539, -3.7632816, -7.1633539, -3.7632816, -2.7387900, 2.7399797
3: -12.8481083, -9.6415062, -12.8481083, -9.6415062, -2.5323596, 2.5334220
4: -6.9691410, -3.4091752, -6.9691410, -3.4091752, -2.9069195, 2.9011769
5: -2.8902202, 0.0193191, -2.8902202, 0.0193191, -2.7060213, 2.7003098
6: 8.6571074, 12.1110544, 8.6571074, 12.1110544, -3.1612616, 3.1630201
7: -18.6063366, -15.0106993, -18.6063366, -15.0106993, -2.6471314, 2.6486826
8: -1.4227927, 1.5777073, -1.4227927, 1.5777073, -2.5498438, 2.5483365
9: -16.1397839, -12.4484825, -16.1397839, -12.4484825, -3.1548033, 3.1524944

Time for backsubstitution: 14.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 805
type: RSZ, layer: 1, pos: 803
type: RSZ, layer: 1, pos: 6185
type: RSZ, layer: 1, pos: 6199
type: RSZ, layer: 1, pos: 5847
type: RSZ, layer: 1, pos: 515
type: RSZ, layer: 1, pos: 4645
type: RSZ, layer: 1, pos: 482
type: RSZ, layer: 1, pos: 110
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 805

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -1.4773688, upper bound: 1.4776773
time: 7.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.4632758, upper bound: 1.4917171
time: 6.55 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 29.16 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.16
Output dim: 6, lower bound: -1.4773690, upper bound: 1.4789225
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.16
Output dim: 6, lower bound: -1.4632758, upper bound: 1.4929515
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.16
Output dim: 6, lower bound: -1.4773688, upper bound: 1.4776773
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.16
Output dim: 6, lower bound: -1.4632758, upper bound: 1.4917171
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 6, lower bound: -1.4917205, upper bound: 1.4785449
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 6, lower bound: -1.4917204, upper bound: 1.4773767
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 6, lower bound: -1.4773768, upper bound: 1.4917223
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 6, lower bound: -1.4785430, upper bound: 1.4917203
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 6, lower bound: -1.4917228, upper bound: 1.4773751
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.16
Output dim: 6, lower bound: -1.4929555, upper bound: 1.4773746
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=3.1755638122558594
rel_dist={6: [-1.493016446904722, 1.493015957466044]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2436.34 seconds
