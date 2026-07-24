## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist_conv_exp.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 3600 seconds
Threshold: 0.68009896801
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.7687196, 1.7687196)
1: (-17.9640560, -15.6267805, -17.9640560, -15.6267805, -2.3372755, 2.3372755)
2: (-6.5349898, -4.4621940, -6.5349898, -4.4621940, -2.0727959, 2.0727959)
3: (-13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.8710003, 1.8710003)
4: (-5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575)
5: (-7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.4649839, 1.4649839)
6: (8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.7894077, 1.7894077)
7: (-14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.8923044, 1.8923044)
8: (-6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.4643955, 1.4643955)
9: (-10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.3401470, 2.3401470)

## BASE Result
execution time: IAR + LP analysis = 15.11 + 32.16 = 47.27 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.73 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=1.5314500331878662
rel_dist={6: [-1.0446957286966772, 1.0446977795075227]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=1.2817935943603516
rel_dist={6: [-0.7326686905779951, 0.7326686039477615]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=1.1153554916381836
rel_dist={6: [-0.4263656502178641, 0.4263650218753696]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.VERIFIED, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=1.1985745429992676
rel_dist={6: [-0.5968935179809218, 0.5968955062752901]}

## Binary Search Result
Binary search time: 209.49 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_random_Z) starts
Time budget: 3343.25 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1243276, upper bound: 1.1308761
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1308758, upper bound: 1.1243276
time: 3.26 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.02
Output dim: 6, lower bound: -1.1243276, upper bound: 1.1308761
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.02
Output dim: 6, lower bound: -1.1308758, upper bound: 1.1243276

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6853938, 1.6860256
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9728360, 1.9717119
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7684464, 1.7676373
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6117487, 1.6126127
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3117981, 1.3118007
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6120815, 1.6124966
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4668877, 1.4613771
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1305258, 1.1307957
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1465197, 2.1472082

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1515

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 886

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1173189, upper bound: 1.1217941
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1152170, upper bound: 1.1239097
time: 3.69 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6860261, 1.6853938
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9717116, 1.9728363
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7676368, 1.7684469
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6126127, 1.6117492
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3118010, 1.3117983
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6124964, 1.6120815
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4613774, 1.4668877
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1307960, 1.1305258
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1472082, 2.1465201

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2461

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2349

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1308486, upper bound: 1.1192566
time: 3.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1258361, upper bound: 1.1243007
time: 3.65 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 12.55 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.55
Output dim: 6, lower bound: -1.1173189, upper bound: 1.1217941
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.55
Output dim: 6, lower bound: -1.1152170, upper bound: 1.1239097
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 12.55
Output dim: 6, lower bound: -1.1308486, upper bound: 1.1192566
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 12.55
Output dim: 6, lower bound: -1.1258361, upper bound: 1.1243007

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6870017, 1.6860251
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9728360, 1.9702282
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7670999, 1.7672820
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6099710, 1.6126122
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3120208, 1.3116765
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6112804, 1.6094623
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4668868, 1.4621689
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1324549, 1.1307956
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1449208, 2.1472068

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1502

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2461

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1101978, upper bound: 1.1216100
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1171292, upper bound: 1.1147001
time: 3.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6853933, 1.6860256
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9728360, 1.9717112
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7680917, 1.7676373
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6117487, 1.6126127
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3116741, 1.3118007
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6120815, 1.6116958
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4668877, 1.4613769
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1305256, 1.1307957
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1465192, 2.1472082

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1406

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1152155, upper bound: 1.1226933
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1140029, upper bound: 1.1239097
time: 3.88 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6864762, 1.6842766
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9746146, 1.9765306
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7715087, 1.7706976
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6125379, 1.6115928
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3089840, 1.3099713
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6112924, 1.6108980
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4529438, 1.4545493
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1316819, 1.1307107
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1464024, 2.1456561

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1075484, upper bound: 1.0968898
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1075484, upper bound: 1.0968898
time: 3.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6849089, 1.6858439
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9754057, 1.9757392
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7698879, 1.7723188
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6124563, 1.6116743
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3099735, 1.3089814
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6113133, 1.6108773
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4490390, 1.4584544
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1309807, 1.1314120
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1463442, 2.1457138

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1479

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1122880, upper bound: 1.1062341
time: 3.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1063336, upper bound: 1.1090842
time: 4.01 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 12.94 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 6, lower bound: -1.1101978, upper bound: 1.1216100
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 6, lower bound: -1.1171292, upper bound: 1.1147001
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 6, lower bound: -1.1152155, upper bound: 1.1226933
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 6, lower bound: -1.1140029, upper bound: 1.1239097
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 6, lower bound: -1.1075484, upper bound: 1.0968898
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 6, lower bound: -1.1075484, upper bound: 1.0968898
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 6, lower bound: -1.1122880, upper bound: 1.1062341
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 12.94
Output dim: 6, lower bound: -1.1063336, upper bound: 1.1090842

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6908007, 1.6870809
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9727168, 1.9700017
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7636161, 1.7645521
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5977697, 1.6036758
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3116734, 1.3114243
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6087055, 1.6062799
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4713650, 1.4679909
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1260200, 1.1251473
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1434913, 2.1455932

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1479

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0949447, upper bound: 1.1020812
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0921225, upper bound: 1.1080635
time: 3.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6880574, 1.6898241
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9726090, 1.9701092
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7643700, 1.7637982
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6010346, 1.6004109
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3117688, 1.3113294
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6080980, 1.6068871
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4727087, 1.4666464
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1268065, 1.1243606
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1433063, 2.1457772

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1979

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1847

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1161951, upper bound: 1.1143949
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1168235, upper bound: 1.1137666
time: 3.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6853952, 1.6860251
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9728489, 1.9717238
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7680907, 1.7676387
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6117911, 1.6126294
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3117552, 1.3118470
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6120734, 1.6116872
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4668055, 1.4613118
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1305835, 1.1308669
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1465130, 2.1472082

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2356

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1031317, upper bound: 1.1114567
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1035541, upper bound: 1.1101143
time: 3.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6853924, 1.6860280
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9728494, 1.9717236
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7680931, 1.7676363
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6117654, 1.6126556
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3117204, 1.3118818
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6120729, 1.6116879
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4668226, 1.4612947
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1305964, 1.1308538
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1465187, 2.1472025

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1506

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2130

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0704633, upper bound: 1.0850490
time: 3.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0770043, upper bound: 1.0795379
time: 3.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6861916, 1.6833577
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9715619, 1.9707170
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7668271, 1.7661057
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6083903, 1.6094475
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3077378, 1.3008857
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5912118, 1.5965376
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4458489, 1.4428644
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1159916, 1.1383038
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1433902, 2.1291203

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1479

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0939520, upper bound: 1.0788418
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0879730, upper bound: 1.0816600
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6864762, 1.6839924
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9688010, 1.9765306
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7715087, 1.7660160
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6125379, 1.6074452
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2998986, 1.3099713
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5969319, 1.6108980
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4529438, 1.4474542
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1316819, 1.1150203
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1298671, 2.1456561

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2461

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1016190, upper bound: 1.0966799
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1073669, upper bound: 1.0929581
time: 3.72 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6771045, 1.6765842
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9536529, 1.9460943
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7700729, 1.7725830
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6020055, 1.5871968
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2659478, 1.2628038
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5730796, 1.5660322
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4476662, 1.4572995
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1100203, 1.1148438
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1349516, 2.1340079

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2130

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0674570, upper bound: 1.0690718
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0723516, upper bound: 1.0625247
time: 3.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6756496, 1.6780391
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9457612, 1.9539859
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7701521, 1.7725034
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5879788, 1.6012235
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2637963, 1.2616136
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5664678, 1.5726438
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4478836, 1.4570818
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1144124, 1.1104517
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1335983, 2.1343212

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1705

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0926902, upper bound: 1.0946655
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0926902, upper bound: 1.0946655
time: 3.89 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.47 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.0949447, upper bound: 1.1020812
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.0921225, upper bound: 1.1080635
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.1161951, upper bound: 1.1143949
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.1168235, upper bound: 1.1137666
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.1031317, upper bound: 1.1114567
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.1035541, upper bound: 1.1101143
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.0704633, upper bound: 1.0850490
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.0770043, upper bound: 1.0795379
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.0939520, upper bound: 1.0788418
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.0879730, upper bound: 1.0816600
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.1016190, upper bound: 1.0966799
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.1073669, upper bound: 1.0929581
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.0674570, upper bound: 1.0690718
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.0723516, upper bound: 1.0625247
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.0926902, upper bound: 1.0946655
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.47
Output dim: 6, lower bound: -1.0926902, upper bound: 1.0946655

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6829958, 1.6778216
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9509645, 1.9403572
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7638006, 1.7648163
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5873194, 1.5791993
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2643056, 1.2652466
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5704718, 1.5614345
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4699917, 1.4668353
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1050587, 1.1085784
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1320992, 2.1328478

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2480

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1997

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0787980, upper bound: 1.0855404
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0787980, upper bound: 1.0855404
time: 3.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6815414, 1.6792765
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9430728, 1.9482491
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7638798, 1.7647367
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5732927, 1.5932260
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2654958, 1.2673984
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5638599, 1.5680461
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4702091, 1.4666178
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1094511, 1.1041862
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1317863, 2.1342015

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 780

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 890

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0912678, upper bound: 1.1074061
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0912818, upper bound: 1.1070238
time: 3.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6847391, 1.6875844
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9734397, 1.9699445
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7609901, 1.7553816
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6004019, 1.6000972
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3054204, 1.3082366
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6080165, 1.6070805
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4712057, 1.4650173
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1261628, 1.1234933
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1314435, 2.1357994

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 890

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1775

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 2461

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0989850, upper bound: 1.1077829
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1090357, upper bound: 1.0979543
time: 3.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6858177, 1.6865058
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9724445, 1.9709396
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7559538, 1.7604189
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6007209, 1.5997782
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3086758, 1.3049810
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6082916, 1.6068051
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4710803, 1.4651432
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1259394, 1.1237168
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1333289, 2.1339140

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 890

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0938112, upper bound: 1.0916201
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0938112, upper bound: 1.0916201
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6905198, 1.6842561
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9719038, 1.9702160
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7550459, 1.7610269
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6100926, 1.6108027
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2874820, 1.2965410
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6081991, 1.6060677
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4637930, 1.4552069
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1270915, 1.1287205
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1493425, 2.1440187

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 718

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0776701, upper bound: 1.0859764
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0776701, upper bound: 1.0859764
time: 3.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6836252, 1.6911507
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9713411, 1.9707789
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7613134, 1.7545938
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6099639, 1.6109314
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2964494, 1.2870269
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6063147, 1.6078129
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4607012, 1.4582992
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1284372, 1.1273749
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1433249, 2.1487203

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1979

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1003714, upper bound: 1.1054694
time: 3.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0988933, upper bound: 1.1068793
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6845503, 1.6849566
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9607563, 1.9593742
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7615457, 1.7465696
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6021495, 1.5697217
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3070383, 1.3163600
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6015272, 1.6149392
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4670146, 1.4610038
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1294315, 1.1310041
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1533670, 2.1460562

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2827

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0578606, upper bound: 1.0728023
time: 3.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0582591, upper bound: 1.0721854
time: 3.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6843209, 1.6860280
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9728494, 1.9596307
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7470269, 1.7676363
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5688310, 1.6126556
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3117204, 1.3072000
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6120729, 1.6011419
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4665315, 1.4612947
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1305964, 1.1296887
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1453724, 2.1472025

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2396

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0640688, upper bound: 1.0633671
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0608985, upper bound: 1.0665024
time: 3.56 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6783872, 1.6740985
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9498091, 1.9410725
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7670116, 1.7663698
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5979395, 1.5849700
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2637115, 1.2547081
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5529780, 1.5516920
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4444766, 1.4417098
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.0950309, 1.1217353
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1319971, 2.1174150

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2396

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0569166, upper bound: 1.0419050
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0569166, upper bound: 1.0419050
time: 3.75 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6769323, 1.6755528
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9419174, 1.9489641
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7670913, 1.7662902
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5839128, 1.5989966
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2615597, 1.2535176
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5463662, 1.5583038
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4446945, 1.4414921
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.0994231, 1.1173432
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1306438, 2.1177278

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1228

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0870534, upper bound: 1.0808732
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0871865, upper bound: 1.0807493
time: 4.14 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6902733, 1.6850467
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9686823, 1.9763041
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7680264, 1.7632008
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6003361, 1.5986209
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2997007, 1.3098000
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5941949, 1.6077147
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4574211, 1.4533539
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1252478, 1.1095200
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1282802, 2.1439686

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 886

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2827

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1002330, upper bound: 1.0945298
time: 4.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0994725, upper bound: 1.0952929
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6875300, 1.6880350
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9685836, 1.9764116
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7687798, 1.7624469
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6036015, 1.5953555
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2997961, 1.3097050
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5935879, 1.6083219
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4587657, 1.4520097
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1260346, 1.1087332
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1280951, 2.1441526

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1705

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2305

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0889195, upper bound: 1.0778742
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0922597, upper bound: 1.0745203
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6762834, 1.6755128
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9411969, 1.9352331
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7635894, 1.7515168
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5923915, 1.5442643
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2612662, 1.2722795
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5625205, 1.5693350
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4478583, 1.4570086
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1088592, 1.1150153
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1421480, 2.1328616

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 738

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1228

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0665464, upper bound: 1.0682855
time: 4.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0666706, upper bound: 1.0681702
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6760325, 1.6765842
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9536529, 1.9336381
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7490063, 1.7725830
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5590730, 1.5871968
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2659478, 1.2581220
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5730796, 1.5554729
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4473758, 1.4572995
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1100203, 1.1136827
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1338053, 2.1340079

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2827

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0721613, upper bound: 1.0608201
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0706485, upper bound: 1.0623343
time: 3.66 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6625595, 1.6707764
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9282818, 1.9246788
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7735701, 1.7624998
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5821314, 1.6012654
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2538285, 1.2747424
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5646605, 1.5715032
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4347129, 1.4409289
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1077744, 1.1079215
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1320581, 2.1365566

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 660

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0769329, upper bound: 1.0916997
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0896419, upper bound: 1.0804353
time: 3.76 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6683865, 1.6780391
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9457612, 1.9365067
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7601490, 1.7725034
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5879788, 1.5953760
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2637963, 1.2516458
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5653272, 1.5726438
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4317307, 1.4570818
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1144124, 1.1038136
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1335983, 2.1327806

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2461

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0584965, upper bound: 1.0604013
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0584965, upper bound: 1.0604013
time: 3.28 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 12.01 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0787980, upper bound: 1.0855404
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0787980, upper bound: 1.0855404
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0912678, upper bound: 1.1074061
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0912818, upper bound: 1.1070238
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0989850, upper bound: 1.1077829
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.1090357, upper bound: 1.0979543
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0938112, upper bound: 1.0916201
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0938112, upper bound: 1.0916201
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0776701, upper bound: 1.0859764
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0776701, upper bound: 1.0859764
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.1003714, upper bound: 1.1054694
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0988933, upper bound: 1.1068793
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0578606, upper bound: 1.0728023
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0582591, upper bound: 1.0721854
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0640688, upper bound: 1.0633671
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0608985, upper bound: 1.0665024
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0569166, upper bound: 1.0419050
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0569166, upper bound: 1.0419050
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0870534, upper bound: 1.0808732
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0871865, upper bound: 1.0807493
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.1002330, upper bound: 1.0945298
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0994725, upper bound: 1.0952929
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0889195, upper bound: 1.0778742
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0922597, upper bound: 1.0745203
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0665464, upper bound: 1.0682855
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0666706, upper bound: 1.0681702
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0721613, upper bound: 1.0608201
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0706485, upper bound: 1.0623343
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0769329, upper bound: 1.0916997
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0896419, upper bound: 1.0804353
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0584965, upper bound: 1.0604013
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 12.01
Output dim: 6, lower bound: -1.0584965, upper bound: 1.0604013

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6833982, 1.6770592
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9367580, 1.9482703
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7624607, 1.7648702
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5821772, 1.5821171
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2629266, 1.2654288
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5704031, 1.5612221
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4683275, 1.4687641
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1044481, 1.1079724
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1319952, 2.1328158

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 738

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1515

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0708317, upper bound: 1.0852781
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0785358, upper bound: 1.0730084
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6822338, 1.6778216
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9509645, 1.9261513
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7638006, 1.7634768
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5873194, 1.5740571
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2643056, 1.2638676
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5702591, 1.5614345
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4699917, 1.4651716
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1050587, 1.1079676
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1320992, 2.1327438

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 332

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1847

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0779115, upper bound: 1.0852401
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0784922, upper bound: 1.0845353
time: 3.88 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6801577, 1.6777053
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9427295, 1.9473557
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7632627, 1.7641554
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5722752, 1.5922542
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2635255, 1.2674372
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5625377, 1.5666742
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4629121, 1.4540248
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1090282, 1.1037025
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1290979, 2.1294651

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1847

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0736009, upper bound: 1.1004757
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0845814, upper bound: 1.0921104
time: 3.97 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6799703, 1.6778927
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9421792, 1.9479060
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7632990, 1.7641191
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5723209, 1.5922084
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2640271, 1.2654278
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5624881, 1.5667238
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4576163, 1.4591067
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1089677, 1.1037636
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1270504, 2.1301756

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 332

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0750268, upper bound: 1.0899316
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0742861, upper bound: 1.0906218
time: 5.21 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6826615, 1.6847882
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9722548, 1.9711320
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7606430, 1.7550416
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6013398, 1.5952454
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3052688, 1.3081825
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6077499, 1.6071787
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4724863, 1.4628072
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1223722, 1.1217146
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1314769, 2.1356511

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 913

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2917

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0548366, upper bound: 1.0628023
time: 3.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0548366, upper bound: 1.0628023
time: 3.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6819434, 1.6875844
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9734397, 1.9687598
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7606502, 1.7553816
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5955501, 1.6000972
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3053665, 1.3082366
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6080165, 1.6068141
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4689953, 1.4650173
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1243844, 1.1234933
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1312947, 2.1357994

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0851876, upper bound: 1.0752919
time: 3.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0851876, upper bound: 1.0752919
time: 3.95 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6855345, 1.6855111
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9693928, 1.9651270
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7511883, 1.7557445
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5966854, 1.5977449
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3075514, 1.2960169
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5880599, 1.5922840
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4640627, 1.4535360
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1103957, 1.1314570
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1302328, 2.1172943

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 780

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0801021, upper bound: 1.0709930
time: 3.58 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0748818, upper bound: 1.0780965
time: 4.01 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6858177, 1.6862221
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9666324, 1.9709396
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7559538, 1.7556529
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6007209, 1.5957422
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2997122, 1.3049810
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5937705, 1.6068051
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4710803, 1.4581258
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1259394, 1.1081733
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1167088, 2.1339140

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1705

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0744291, upper bound: 1.0727712
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0745305, upper bound: 1.0727715
time: 3.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6875749, 1.6893463
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9697318, 1.9713798
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7928891, 1.7506285
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6057434, 1.6289520
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2856615, 1.2991235
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6063919, 1.6054492
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4661088, 1.4534841
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1224025, 1.1244124
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1489224, 2.1437664

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1979

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0739210, upper bound: 1.0809237
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0726199, upper bound: 1.0821828
time: 4.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6905198, 1.6813111
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9719038, 1.9680440
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7446475, 1.7610269
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6100926, 1.6064534
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2874820, 1.2947209
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6075807, 1.6060677
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4620700, 1.4552069
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1270915, 1.1240315
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1490884, 2.1440187

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 151

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2917

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0309823, upper bound: 1.0395491
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0309823, upper bound: 1.0395491
time: 3.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6833577, 1.6898808
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9706459, 1.9706976
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7610784, 1.7545090
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6096811, 1.6107244
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2961357, 1.2873056
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6050124, 1.6056421
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4597149, 1.4579341
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1282159, 1.1269845
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1415691, 2.1472249

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 613

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.1000385, upper bound: 1.1021109
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0970105, upper bound: 1.1051289
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6823549, 1.6911507
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9712601, 1.9707789
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7612290, 1.7545938
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6097569, 1.6109314
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2964494, 1.2867129
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6063147, 1.6065111
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4603357, 1.4582992
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1284372, 1.1271536
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1418285, 2.1487203

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1228

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0980692, upper bound: 1.1060751
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0980890, upper bound: 1.1060208
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6896806, 1.6831870
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9601855, 1.9582405
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7478793, 1.7393370
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6005468, 1.5679913
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2835853, 1.3018742
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5960932, 1.6053274
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4639926, 1.4548893
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1257561, 1.1287265
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1530180, 2.1428676

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2356

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 1228

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0570366, upper bound: 1.0720001
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0570562, upper bound: 1.0719463
time: 3.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6827807, 1.6900816
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9596224, 1.9587796
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7539566, 1.7329035
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6004181, 1.5681200
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2925527, 1.2950566
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5942087, 1.6095057
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4609008, 1.4579818
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1271018, 1.1273288
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1501780, 2.1475692

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 738

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 613

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0578668, upper bound: 1.0687947
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0548704, upper bound: 1.0717909
time: 3.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6843758, 1.6853371
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9719801, 1.9584591
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7476330, 1.7664080
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5686460, 1.6127720
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3097792, 1.3053317
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6118946, 1.6009557
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4662642, 1.4612710
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1305947, 1.1296501
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1451097, 2.1471934

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 577

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1705

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0500772, upper bound: 1.0493364
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0500772, upper bound: 1.0493362
time: 3.59 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6836295, 1.6860280
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9716773, 1.9596307
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7457991, 1.7676363
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5688310, 1.6124706
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3117204, 1.3052583
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6118870, 1.6011419
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4665074, 1.4612947
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1305964, 1.1296868
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1453633, 2.1472025

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 660

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1979

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0576326, upper bound: 1.0618177
time: 3.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0562317, upper bound: 1.0633122
time: 3.70 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6689854, 1.6729107
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9424701, 1.9321136
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7643166, 1.7683053
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5974774, 1.5850739
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2611616, 1.2528169
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5547574, 1.5515695
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4428806, 1.4478970
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.0955430, 1.1189977
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1316099, 2.1190295

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2827

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 332

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0406007, upper bound: 1.0245265
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0407889, upper bound: 1.0245797
time: 4.00 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6771998, 1.6740985
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9498091, 1.9337332
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7670116, 1.7636747
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5979395, 1.5845079
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2618203, 1.2547081
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5528553, 1.5516920
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4444766, 1.4401135
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.0922931, 1.1217353
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1319971, 2.1170278

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2461

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0471928, upper bound: 1.0315658
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0472147, upper bound: 1.0315452
time: 3.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6761255, 1.6749678
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9421244, 1.9490609
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7625017, 1.7635369
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5841212, 1.6006355
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2640262, 1.2542443
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5398767, 1.5518725
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4453738, 1.4423981
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1005218, 1.1179109
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1286163, 2.1156354

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2563

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 332

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0709508, upper bound: 1.0639049
time: 4.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0711379, upper bound: 1.0639310
time: 5.08 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6763468, 1.6747460
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9416933, 1.9491711
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7643380, 1.7617006
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5855517, 1.5992055
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2622895, 1.2559841
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5399349, 1.5518143
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4456003, 1.4421718
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.0999908, 1.1184418
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1285515, 2.1156812

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 330

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1406

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0871850, upper bound: 1.0795037
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0859390, upper bound: 1.0807478
time: 5.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6902719, 1.6850462
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9686823, 1.9763026
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7680211, 1.7631969
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6003356, 1.5986195
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2997007, 1.3097997
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5941834, 1.6076965
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4574101, 1.4533482
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1252363, 1.1095126
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1282773, 2.1439672

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1515

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0999829, upper bound: 1.0904025
time: 4.12 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0982549, upper bound: 1.0942345
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6902728, 1.6850467
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9686813, 1.9763041
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7680221, 1.7632008
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6003361, 1.5986199
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2997007, 1.3098000
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5941949, 1.6077027
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4574153, 1.4533539
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1252403, 1.1095200
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1282802, 2.1439662

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1979

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0961537, upper bound: 1.0906642
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0948453, upper bound: 1.0919726
time: 4.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6875305, 1.6880350
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9685831, 1.9764102
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7687798, 1.7624469
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6036015, 1.5953565
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2997961, 1.3097054
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5935874, 1.6083231
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4587669, 1.4520094
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1260346, 1.1087333
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1280975, 2.1441517

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 913

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1479

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0750859, upper bound: 1.0596459
time: 3.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0694746, upper bound: 1.0629005
time: 3.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6875305, 1.6880350
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9685836, 1.9764109
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7687798, 1.7624469
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6036015, 1.5953560
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2997961, 1.3097049
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5935879, 1.6083214
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4587655, 1.4520097
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1260346, 1.1087337
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1280947, 2.1441526

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1705

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0727729, upper bound: 1.0562604
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0728738, upper bound: 1.0562603
time: 3.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6754775, 1.6749287
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9414043, 1.9350100
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7589989, 1.7487636
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5925999, 1.5459032
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2637322, 1.2730093
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5560308, 1.5629032
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4485378, 1.4579146
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1099575, 1.1155825
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1401014, 2.1307693

Time for backsubstitution: 5.50 seconds
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.614668846130371
rel_dist={6: [-1.1326113226070742, 1.1326113586240112]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2349

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8464561, upper bound: 0.8441697
time: 7.72 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8441699, upper bound: 0.8464563
time: 10.25 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.99 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 17.99
Output dim: 6, lower bound: -0.8464561, upper bound: 0.8441697
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 17.99
Output dim: 6, lower bound: -0.8441699, upper bound: 0.8464563

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5096412, 1.5087457
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5895929, 1.5900450
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4742408, 1.4733148
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3395061, 1.3394594
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8541317, 1.8554850
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0688872, 1.0694530
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3638158, 1.3638277
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1540940, 1.1518629
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8982143, 0.8978134
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8535728, 1.8535399

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 886

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8431820, upper bound: 0.8360217
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8368505, upper bound: 0.8409602
time: 7.79 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5087457, 1.5096412
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5900450, 1.5895929
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4733148, 1.4742413
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3394594, 1.3395061
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8554850, 1.8541317
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0694530, 1.0688874
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3638277, 1.3638160
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1518633, 1.1540945
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8978136, 0.8982142
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8535404, 1.8535728

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 414

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 615

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8441683, upper bound: 0.8463941
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8441150, upper bound: 0.8464565
time: 4.45 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 16.59 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.59
Output dim: 6, lower bound: -0.8431820, upper bound: 0.8360217
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.59
Output dim: 6, lower bound: -0.8368505, upper bound: 0.8409602
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 16.59
Output dim: 6, lower bound: -0.8441683, upper bound: 0.8463941
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 16.59
Output dim: 6, lower bound: -0.8441150, upper bound: 0.8464565

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5083852, 1.5083160
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5815635, 1.5823474
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4643173, 1.4573498
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3250451, 1.3249917
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8225346, 1.8325272
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0557783, 1.0693319
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3860989, 1.3750668
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1651714, 1.1648262
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8966455, 0.8916445
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8448548, 1.8478394

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2396

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8335317, upper bound: 0.8219370
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8316128, upper bound: 0.8261824
time: 3.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5088086, 1.5074897
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5813637, 1.5820158
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4582758, 1.4633913
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3249936, 1.3249984
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8311729, 1.8238888
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0687666, 1.0563438
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3750553, 1.3861108
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1670578, 1.1629400
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8920455, 0.8962449
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8478723, 1.8448219

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2917

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8295819, upper bound: 0.8355213
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8312548, upper bound: 0.8354292
time: 5.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5087485, 1.5096445
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5900607, 1.5896077
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4732604, 1.4741817
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3395214, 1.3395543
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8554878, 1.8541412
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0694373, 1.0688658
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3637953, 1.3637757
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1517439, 1.1539955
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8977842, 0.8981918
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8534470, 1.8534789

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8385023, upper bound: 0.8407628
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8386363, upper bound: 0.8406709
time: 5.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5087490, 1.5096440
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5900602, 1.5896087
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4732556, 1.4741869
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3395076, 1.3395681
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8554955, 1.8541341
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0694313, 1.0688717
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3637877, 1.3637836
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1517639, 1.1539755
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8977911, 0.8981850
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8534470, 1.8534803

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 332

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8412198, upper bound: 0.8463006
time: 4.14 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8439463, upper bound: 0.8420177
time: 4.65 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.25 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.25
Output dim: 6, lower bound: -0.8335317, upper bound: 0.8219370
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.25
Output dim: 6, lower bound: -0.8316128, upper bound: 0.8261824
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.25
Output dim: 6, lower bound: -0.8295819, upper bound: 0.8355213
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.25
Output dim: 6, lower bound: -0.8312548, upper bound: 0.8354292
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.25
Output dim: 6, lower bound: -0.8385023, upper bound: 0.8407628
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.25
Output dim: 6, lower bound: -0.8386363, upper bound: 0.8406709
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.25
Output dim: 6, lower bound: -0.8412198, upper bound: 0.8463006
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.25
Output dim: 6, lower bound: -0.8439463, upper bound: 0.8420177

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5081196, 1.5076241
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5805659, 1.5811765
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4631720, 1.4561214
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3248606, 1.3248610
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8223038, 1.8325143
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0544436, 1.0680387
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3859177, 1.3748817
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1650076, 1.1648023
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8966436, 0.8916217
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8447008, 1.8478308

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 780

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8292674, upper bound: 0.8217140
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8333488, upper bound: 0.8177834
time: 8.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5076933, 1.5080504
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5803928, 1.5813494
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4630890, 1.4571695
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3250332, 1.3248076
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8225222, 1.8322964
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0544856, 1.0679975
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3859134, 1.3748856
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1651478, 1.1646626
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8966227, 0.8916426
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8448458, 1.8476853

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 718

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8217896, upper bound: 0.8244920
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8299658, upper bound: 0.8163131
time: 3.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5092225, 1.5055695
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5803509, 1.5801854
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4568467, 1.4654016
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3243365, 1.3239822
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8293200, 1.8235421
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0664914, 1.0530756
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3746638, 1.3852968
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1643107, 1.1598239
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8861341, 0.8919413
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8474736, 1.8446445

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2827

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8295692, upper bound: 0.8346148
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8286751, upper bound: 0.8355104
time: 3.62 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5068884, 1.5074897
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5813637, 1.5810030
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4582758, 1.4619622
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3249936, 1.3243413
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8308268, 1.8238888
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0687666, 1.0540686
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3750553, 1.3857193
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1639416, 1.1629400
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8920455, 0.8903339
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8476949, 1.8448219

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2356

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1775

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8267844, upper bound: 0.8350981
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8309221, upper bound: 0.8309177
time: 5.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5099053, 1.5082684
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5890756, 1.5878048
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4718132, 1.4787426
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3388062, 1.3384805
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8537617, 1.8537941
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0664306, 1.0648661
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3634048, 1.3629620
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1507678, 1.1510124
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8934047, 0.8974723
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8528681, 1.8531213

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2480

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 890

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8379545, upper bound: 0.8401692
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8379047, upper bound: 0.8402201
time: 5.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5073724, 1.5096445
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5900607, 1.5886226
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4732604, 1.4727345
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3395214, 1.3388391
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8551407, 1.8541412
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0694373, 1.0658591
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3637953, 1.3633852
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1487603, 1.1539955
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8977842, 0.8938124
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8530893, 1.8534789

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8355200, upper bound: 0.8295453
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8308141, upper bound: 0.8375417
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5115113, 1.5106988
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5899420, 1.5894341
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4700947, 1.4714570
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3273077, 1.3292336
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8537397, 1.8522329
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0692060, 1.0687008
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3612127, 1.3608618
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1562424, 1.1592219
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8913568, 0.8922004
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8518634, 1.8517919

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2356

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3102

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8374628, upper bound: 0.8449629
time: 7.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8398506, upper bound: 0.8425683
time: 4.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5098038, 1.5122662
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5898805, 1.5894907
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4705257, 1.4710264
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3291731, 1.3273678
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8535938, 1.8523784
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0692604, 1.0686464
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3608656, 1.3612087
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1570101, 1.1584537
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8918062, 0.8917508
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8517575, 1.8518968

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 330

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8437560, upper bound: 0.8410714
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8429953, upper bound: 0.8418253
time: 4.04 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.11 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8292674, upper bound: 0.8217140
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8333488, upper bound: 0.8177834
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8217896, upper bound: 0.8244920
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8299658, upper bound: 0.8163131
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8295692, upper bound: 0.8346148
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8286751, upper bound: 0.8355104
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8267844, upper bound: 0.8350981
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8309221, upper bound: 0.8309177
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8379545, upper bound: 0.8401692
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8379047, upper bound: 0.8402201
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8355200, upper bound: 0.8295453
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8308141, upper bound: 0.8375417
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8374628, upper bound: 0.8449629
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8398506, upper bound: 0.8425683
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8437560, upper bound: 0.8410714
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.11
Output dim: 6, lower bound: -0.8429953, upper bound: 0.8418253

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5107551, 1.5086923
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5804367, 1.5809460
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4611006, 1.4542122
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3120565, 1.3145504
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8202686, 1.8303332
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0540879, 1.0677373
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3833499, 1.3719664
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1695056, 1.1706533
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8887901, 0.8842176
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8429632, 1.8461366

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2858

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8206584, upper bound: 0.8117871
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8192182, upper bound: 0.8131079
time: 6.17 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5091877, 1.5104618
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5803800, 1.5810473
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4612617, 1.4537811
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3139224, 1.3120570
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8201237, 1.8305459
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0541422, 1.0675349
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3831110, 1.3723135
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1702738, 1.1692996
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8891120, 0.8837681
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8430071, 1.8462415

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 151

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8316744, upper bound: 0.8123502
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8278935, upper bound: 0.8161620
time: 5.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5053072, 1.5052533
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5792079, 1.5815203
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4627452, 1.4568300
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3234892, 1.3199553
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8196220, 1.8294187
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0543756, 1.0679433
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3856463, 1.3748274
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1649327, 1.1624532
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8936946, 0.8898640
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8448009, 1.8475370

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2356

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1997

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8109045, upper bound: 0.8140938
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8109045, upper bound: 0.8140938
time: 4.42 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5048966, 1.5056643
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5805635, 1.5801647
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4627495, 1.4568257
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3201809, 1.3232636
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8196440, 1.8293958
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0544314, 1.0678875
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3858547, 1.3746185
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1629386, 1.1644480
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8948443, 0.8887142
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8446980, 1.8476410

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1506

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8297660, upper bound: 0.8161871
time: 3.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8298433, upper bound: 0.8160873
time: 3.73 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5092220, 1.5055695
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5803509, 1.5801845
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4568424, 1.4653978
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3243351, 1.3239808
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8293171, 1.8235397
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0664911, 1.0530751
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3746529, 1.3852825
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1643012, 1.1598179
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8861244, 0.8919337
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8474717, 1.8446422

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1847

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8292264, upper bound: 0.8344370
time: 5.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8293613, upper bound: 0.8324621
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5092225, 1.5055690
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5803499, 1.5801854
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4568434, 1.4653974
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3243346, 1.3239813
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8293171, 1.8235393
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0664911, 1.0530753
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3746490, 1.3852863
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1643050, 1.1598148
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8861268, 0.8919313
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8474717, 1.8446417

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1775

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1695

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8286105, upper bound: 0.8352647
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8284590, upper bound: 0.8354441
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4996877, 1.5013151
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5722594, 1.5734196
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4542947, 1.4573154
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3321867, 1.3293362
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8405781, 1.8366289
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0644422, 1.0504658
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3701482, 1.3815145
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1685014, 1.1664314
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8830036, 0.8816162
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8476834, 1.8436823

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2827

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 780

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8262760, upper bound: 0.8330046
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8249434, upper bound: 0.8346127
time: 6.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5007143, 1.5002880
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5737796, 1.5718994
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4536300, 1.4579802
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3299890, 1.3315339
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8435650, 1.8336411
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0651641, 1.0497439
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3708496, 1.3808126
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1674333, 1.1674995
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8833271, 0.8812928
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8465552, 1.8448110

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1228

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8203185, upper bound: 0.8205469
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8205410, upper bound: 0.8203432
time: 4.61 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5087028, 1.5069590
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5894084, 1.5878372
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4712114, 1.4781618
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3368001, 1.3365002
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8520861, 1.8534803
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0690498, 1.0683635
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3611307, 1.3606601
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1369009, 1.1342416
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8932002, 0.8972325
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8484950, 1.8478570

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 38

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8356528, upper bound: 0.8399629
time: 5.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8377274, upper bound: 0.8362257
time: 5.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5085959, 1.5070662
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5891080, 1.5881381
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4712324, 1.4781408
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3368263, 1.3364744
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8534479, 1.8521180
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0699282, 1.0674853
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3611026, 1.3606882
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1339970, 1.1371455
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8931651, 0.8972673
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8476043, 1.8487477

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1502

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2305

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8281546, upper bound: 0.8332483
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8310265, upper bound: 0.8303760
time: 4.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5055728, 1.5088120
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5820317, 1.5803659
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4633365, 1.4567862
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3250608, 1.3243847
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8235455, 1.8311830
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0563285, 1.0664701
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3860784, 1.3746238
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1597052, 1.1669590
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8962162, 0.8861122
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8445511, 1.8477783

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1732

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2356

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8312041, upper bound: 0.8295451
time: 4.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8355200, upper bound: 0.8252202
time: 5.93 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5063992, 1.5083885
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5823636, 1.5805659
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4572949, 1.4628277
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3250546, 1.3244367
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8321838, 1.8225446
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0693166, 1.0534818
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3750343, 1.3856678
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1615915, 1.1650729
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8916159, 0.8907125
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8475685, 1.8447609

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2901

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8210567, upper bound: 0.8362005
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8292440, upper bound: 0.8279579
time: 3.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5128241, 1.5123725
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5944972, 1.5933473
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4662132, 1.4671125
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3258948, 1.3283148
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8458614, 1.8439531
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0627401, 1.0622362
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3586249, 1.3585110
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1448505, 1.1446810
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8914429, 0.8924408
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8520303, 1.8523526

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2356

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8242458, upper bound: 0.8316292
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8242458, upper bound: 0.8316293
time: 3.99 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5131855, 1.5120115
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5938549, 1.5939898
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4657507, 1.4675756
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3263884, 1.3278213
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8454590, 1.8443556
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0627415, 1.0622349
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3588619, 1.3582737
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1417015, 1.1478298
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8915974, 0.8922865
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8524241, 1.8519592

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 913

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 780

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8393150, upper bound: 0.8378136
time: 5.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8376925, upper bound: 0.8420078
time: 5.51 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5092988, 1.5116291
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5903859, 1.5901515
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4707208, 1.4712811
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3302274, 1.3284736
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8526926, 1.8517284
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0695660, 1.0689672
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3610301, 1.3613648
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1575804, 1.1589150
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8920259, 0.8919812
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8512659, 1.8514261

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1406

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8303210, upper bound: 0.8278897
time: 5.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8303210, upper bound: 0.8278896
time: 6.26 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5091667, 1.5117612
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5905414, 1.5899959
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4707808, 1.4712210
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3302794, 1.3284216
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8529434, 1.8514776
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0695813, 1.0689521
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3610220, 1.3613732
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1574717, 1.1590235
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8920369, 0.8919703
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8512869, 1.8514056

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 738

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8324003, upper bound: 0.8314579
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8326230, upper bound: 0.8312419
time: 5.37 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 16.90 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8206584, upper bound: 0.8117871
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8192182, upper bound: 0.8131079
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8316744, upper bound: 0.8123502
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8278935, upper bound: 0.8161620
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8109045, upper bound: 0.8140938
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8109045, upper bound: 0.8140938
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8297660, upper bound: 0.8161871
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8298433, upper bound: 0.8160873
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8292264, upper bound: 0.8344370
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8293613, upper bound: 0.8324621
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8286105, upper bound: 0.8352647
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8284590, upper bound: 0.8354441
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8262760, upper bound: 0.8330046
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8249434, upper bound: 0.8346127
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8203185, upper bound: 0.8205469
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8205410, upper bound: 0.8203432
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8356528, upper bound: 0.8399629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8377274, upper bound: 0.8362257
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8281546, upper bound: 0.8332483
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8310265, upper bound: 0.8303760
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8312041, upper bound: 0.8295451
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8355200, upper bound: 0.8252202
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8210567, upper bound: 0.8362005
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8292440, upper bound: 0.8279579
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8242458, upper bound: 0.8316292
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8242458, upper bound: 0.8316293
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8393150, upper bound: 0.8378136
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8376925, upper bound: 0.8420078
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8303210, upper bound: 0.8278897
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8303210, upper bound: 0.8278896
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8324003, upper bound: 0.8314579
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 16.90
Output dim: 6, lower bound: -0.8326230, upper bound: 0.8312419

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5110779, 1.5081353
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5802345, 1.5806470
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4602737, 1.4541101
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3095627, 1.3127170
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8198891, 1.8300014
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0536520, 1.0677922
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3838563, 1.3715405
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1683993, 1.1719759
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8899801, 0.8831600
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8404212, 1.8437104

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1775

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3102

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8135589, upper bound: 0.8058940
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8135589, upper bound: 0.8058940
time: 4.40 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5101986, 1.5086923
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5804367, 1.5807440
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4609990, 1.4542122
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3120565, 1.3120565
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8199368, 1.8303332
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0540879, 1.0673015
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3829241, 1.3719664
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1695056, 1.1695473
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8877327, 0.8842176
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8405385, 1.8461366

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1228

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1775

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8149627, upper bound: 0.8127719
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8188828, upper bound: 0.8088192
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5091858, 1.5104594
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5803795, 1.5810454
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4612622, 1.4537826
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3139229, 1.3120570
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8201218, 1.8305449
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0541418, 1.0675340
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3831129, 1.3723140
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1702733, 1.1692996
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8891120, 0.8837681
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8430071, 1.8462424

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 332

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2563

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8100572, upper bound: 0.7905532
time: 4.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8100572, upper bound: 0.7905532
time: 4.36 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5091853, 1.5104604
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5803785, 1.5810461
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4612632, 1.4537816
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3139229, 1.3120570
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8201218, 1.8305449
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0541415, 1.0675344
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3831110, 1.3723154
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1702733, 1.1692996
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8891120, 0.8837682
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8430071, 1.8462415

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 886

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 192

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8171014, upper bound: 0.8075545
time: 6.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8181371, upper bound: 0.8075542
time: 8.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5055680, 1.5044909
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5653758, 1.5804765
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4614248, 1.4563055
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3183460, 1.3200135
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8180876, 1.8261452
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0530865, 1.0675530
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3855085, 1.3746071
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1629553, 1.1626658
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8930948, 0.8892670
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8446970, 1.8474741

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2858

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1705

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8030492, upper bound: 0.8064623
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8030492, upper bound: 0.8064623
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5045443, 1.5052533
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5792079, 1.5676880
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4627452, 1.4555092
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3234892, 1.3148127
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8196220, 1.8278847
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0543756, 1.0666542
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3854265, 1.3748274
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1649327, 1.1604757
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8936946, 0.8892642
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8448009, 1.8474331

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1979

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8093224, upper bound: 0.8113665
time: 6.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8081795, upper bound: 0.8125153
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5048442, 1.5057383
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5963044, 1.5956306
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4623671, 1.4563417
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3181014, 1.3207355
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8239937, 1.8364615
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0553081, 1.0693705
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3859253, 1.3747931
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1452332, 1.1465323
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8971195, 0.8912340
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8447371, 1.8476076

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 890

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1479

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8217178, upper bound: 0.8022638
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8167271, upper bound: 0.8092177
time: 3.99 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5049710, 1.5056119
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5960293, 1.5959053
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4622655, 1.4564433
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3176522, 1.3211846
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8267097, 1.8337455
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0559146, 1.0687642
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3860288, 1.3746896
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1450224, 1.1467435
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8973641, 0.8909894
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8446636, 1.8476806

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1228

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1775

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8254989, upper bound: 0.8157511
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8295079, upper bound: 0.8127401
time: 6.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5059052, 1.5028691
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5807552, 1.5800202
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4513087, 1.4569817
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3237014, 1.3235297
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8293514, 1.8208456
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0601423, 1.0485861
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3745708, 1.3853588
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1627984, 1.1582427
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8854806, 0.8911622
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8356085, 1.8338571

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1515

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8247319, upper bound: 0.8337313
time: 7.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8289866, upper bound: 0.8284350
time: 4.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5065212, 1.5022526
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5801864, 1.5805888
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4484262, 1.4598594
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3238835, 1.3233476
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8266239, 1.8235731
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0620019, 1.0467260
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3747282, 1.3852005
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1627264, 1.1583145
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8853528, 0.8912899
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8366861, 1.8327799

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2480

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1515

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8248747, upper bound: 0.8317567
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8291213, upper bound: 0.8264749
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5069394, 1.5028749
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5781522, 1.5770369
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4574356, 1.4662490
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3238935, 1.3235250
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8318939, 1.8258715
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0648499, 1.0514066
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3724174, 1.3827553
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1448827, 1.1453881
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8864772, 0.8896805
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8463058, 1.8438592

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 332

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 613

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8284745, upper bound: 0.8334261
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8269143, upper bound: 0.8351117
time: 6.13 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5065279, 1.5032864
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5772014, 1.5779874
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4576950, 1.4659901
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3238788, 1.3235397
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8316498, 1.8261151
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0648222, 1.0514340
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3721185, 1.3830543
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1498780, 1.1403928
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8838758, 0.8922818
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8466892, 1.8434763

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 613

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8283220, upper bound: 0.8336047
time: 3.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8267563, upper bound: 0.8352950
time: 4.08 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5016351, 1.5031552
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5672479, 1.5695052
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4504757, 1.4524131
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3320179, 1.3291845
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8428993, 1.8393307
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0623364, 1.0487983
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3691306, 1.3797274
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1675684, 1.1650648
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8832881, 0.8808953
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8442554, 1.8402448

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 151

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8156131, upper bound: 0.8226401
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8157912, upper bound: 0.8224152
time: 4.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5015278, 1.5028706
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5678391, 1.5684080
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4501247, 1.4534960
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3320351, 1.3293295
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8432817, 1.8389482
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0627747, 1.0483601
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3683615, 1.3804960
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1671350, 1.1654985
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8822837, 0.8818995
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8442459, 1.8402534

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 330

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1479

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8170208, upper bound: 0.8216918
time: 6.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8120258, upper bound: 0.8266789
time: 6.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5029321, 1.4985671
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5724840, 1.5702817
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4406004, 1.4486260
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3284378, 1.3299108
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8337793, 1.8230062
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0441430, 1.0341592
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3669124, 1.3757992
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1636314, 1.1619308
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8798349, 0.8785696
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8457174, 1.8405347

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1515

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8157596, upper bound: 0.8198414
time: 4.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8201319, upper bound: 0.8143203
time: 6.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4989924, 1.5025067
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5721622, 1.5706034
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4442768, 1.4449501
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3283644, 1.3299847
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8329325, 1.8238535
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0495796, 1.0287225
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3658357, 1.3768759
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1618643, 1.1636977
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8806040, 0.8778006
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8422794, 1.8439732

Time for backsubstitution: 5.51 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 613

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2234

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8155785, upper bound: 0.8200413
time: 4.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8202321, upper bound: 0.8154100
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5114717, 1.5080152
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5892892, 1.5876629
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4681220, 1.4755030
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3245993, 1.3261652
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8503494, 1.8515782
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0688112, 1.0681791
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3585548, 1.3577366
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1412461, 1.1393547
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8869134, 0.8924074
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8469110, 1.8461680

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2563

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7878252, upper bound: 0.7917709
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7878252, upper bound: 0.7917709
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5097589, 1.5095825
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5892282, 1.5877180
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4685531, 1.4742794
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3264647, 1.3242993
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8501835, 1.8517237
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0688653, 1.0680234
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3582077, 1.3580837
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1420176, 1.1385865
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8873628, 0.8909457
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8468051, 1.8462734

Time for backsubstitution: 5.47 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.3650121688842773
rel_dist={6: [-0.846481615748317, 0.8464836721348625]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 615

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 780

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7321916, upper bound: 0.7297486
time: 7.93 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7297491, upper bound: 0.7321917
time: 7.64 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.58 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.58
Output dim: 6, lower bound: -0.7321916, upper bound: 0.7297486
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.58
Output dim: 6, lower bound: -0.7297491, upper bound: 0.7321917

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4528141, 1.4527335
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4541936, 1.4550164
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3669376, 1.3666754
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2482681, 1.2481589
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7774172, 1.7777042
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9887388, 0.9890674
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2812667, 1.2806902
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0574338, 1.0571089
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8192575, 0.8185042
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7536564, 1.7536492

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2917

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7172236, upper bound: 0.7148472
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7172236, upper bound: 0.7148472
time: 4.17 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4527335, 1.4528141
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4550166, 1.4541934
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3666754, 1.3669381
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2481589, 1.2482681
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7777042, 1.7774172
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9890673, 0.9887388
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2806902, 1.2812672
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0571086, 1.0574341
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8185041, 0.8192575
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7536488, 1.7536559

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 890

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7293853, upper bound: 0.7317831
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7293410, upper bound: 0.7318302
time: 3.88 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.34 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.34
Output dim: 6, lower bound: -0.7172236, upper bound: 0.7148472
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.34
Output dim: 6, lower bound: -0.7172236, upper bound: 0.7148472
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.34
Output dim: 6, lower bound: -0.7293853, upper bound: 0.7317831
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.34
Output dim: 6, lower bound: -0.7293410, upper bound: 0.7318302

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4528046, 1.4527302
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4541473, 1.4550028
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3669300, 1.3666921
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2482519, 1.2481542
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7773924, 1.7776933
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9887390, 0.9890673
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2812657, 1.2806878
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0574002, 1.0571008
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8192437, 0.8184268
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7536411, 1.7536402

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 414

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1979

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7155293, upper bound: 0.7130206
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7153946, upper bound: 0.7131553
time: 4.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4528108, 1.4527335
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4541798, 1.4550164
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3669376, 1.3666668
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2482629, 1.2481589
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7774067, 1.7777042
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9887388, 0.9890676
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2812648, 1.2806902
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0574260, 1.0571089
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8192575, 0.8184905
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7536564, 1.7536349

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1515

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2305

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7098747, upper bound: 0.7103424
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7128149, upper bound: 0.7074463
time: 4.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4515047, 1.4515047
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4553490, 1.4543004
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3660774, 1.3663554
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2461591, 1.2462873
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7760277, 1.7767630
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9916866, 0.9920166
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2784142, 1.2789698
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0425158, 1.0406628
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8182991, 0.8190262
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7490544, 1.7483935

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1705

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2818

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7255197, upper bound: 0.7278767
time: 12.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7254770, upper bound: 0.7279198
time: 7.90 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4514246, 1.4515853
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4551234, 1.4545259
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3660927, 1.3663402
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2461782, 1.2462678
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7770491, 1.7757411
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9923451, 0.9913580
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2783933, 1.2789907
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0403376, 1.0428410
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8182731, 0.8190525
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7483869, 1.7490611

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2396

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7219343, upper bound: 0.7218198
time: 3.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7154006, upper bound: 0.7244451
time: 4.50 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.86 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 6, lower bound: -0.7155293, upper bound: 0.7130206
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 6, lower bound: -0.7153946, upper bound: 0.7131553
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 6, lower bound: -0.7098747, upper bound: 0.7103424
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 6, lower bound: -0.7128149, upper bound: 0.7074463
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 6, lower bound: -0.7255197, upper bound: 0.7278767
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 6, lower bound: -0.7254770, upper bound: 0.7279198
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 6, lower bound: -0.7219343, upper bound: 0.7218198
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.86
Output dim: 6, lower bound: -0.7154006, upper bound: 0.7244451

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4519644, 1.4514604
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4538026, 1.4549212
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3667812, 1.3666077
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2480121, 1.2479472
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7767906, 1.7771778
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9884245, 0.9890072
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2799635, 1.2790132
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0567694, 1.0567358
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8190227, 0.8181332
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7520347, 1.7521439

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1406

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 718

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7097134, upper bound: 0.7078961
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7104418, upper bound: 0.7071788
time: 4.31 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4515347, 1.4518900
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4540658, 1.4546583
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3668456, 1.3665433
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2480450, 1.2479148
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7768764, 1.7770915
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9886785, 0.9887530
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2795911, 1.2793856
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0570354, 1.0564697
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8189502, 0.8182057
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7521453, 1.7520351

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 660

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7126032, upper bound: 0.7070260
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7077377, upper bound: 0.7102825
time: 4.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4528117, 1.4527340
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4541793, 1.4550154
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3669376, 1.3666668
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2482629, 1.2481589
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7774067, 1.7777047
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9887385, 0.9890676
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2812638, 1.2806902
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0574262, 1.0571084
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8192570, 0.8184903
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7536559, 1.7536345

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 886

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2130

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6770374, upper bound: 0.6836424
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6828769, upper bound: 0.6778059
time: 4.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4528117, 1.4527335
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4541798, 1.4550157
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3669376, 1.3666668
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2482629, 1.2481589
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7774067, 1.7777042
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9887388, 0.9890673
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2812648, 1.2806892
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0574262, 1.0571089
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8192575, 0.8184905
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7536559, 1.7536349

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 192

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7011955, upper bound: 0.6988579
time: 6.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7017192, upper bound: 0.6984311
time: 4.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4401340, 1.4398069
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4556336, 1.4548903
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3412356, 1.3424411
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2109904, 1.2102861
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7750006, 1.7779431
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9594121, 0.9650314
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2580147, 1.2587214
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0203600, 1.0175524
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8171842, 0.8179749
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7416615, 1.7425165

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1997

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2234

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7216576, upper bound: 0.7275193
time: 4.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7251592, upper bound: 0.7240118
time: 6.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4398069, 1.4401340
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4559388, 1.4545848
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3421626, 1.3415136
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2101579, 1.2111192
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7772074, 1.7757363
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9647017, 0.9597421
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2581658, 1.2585707
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0194049, 1.0185072
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8172476, 0.8179114
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7431779, 1.7410002

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 886

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1515

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7206198, upper bound: 0.7278001
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7250281, upper bound: 0.7210399
time: 4.78 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4510522, 1.4508934
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4540820, 1.4533548
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3656511, 1.3651123
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2459931, 1.2462120
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7768745, 1.7757287
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9904041, 0.9894484
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2782116, 1.2788057
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0402098, 1.0428174
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8182714, 0.8190351
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7482686, 1.7490520

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2917

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 615

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7219327, upper bound: 0.7218097
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7219262, upper bound: 0.7218160
time: 4.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4507322, 1.4512129
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4539523, 1.4534845
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3648653, 1.3658986
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2461224, 1.2460828
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7770386, 1.7755656
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9904358, 0.9894170
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2782083, 1.2788095
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0403137, 1.0427127
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8182557, 0.8190508
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7483773, 1.7489433

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2349

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7108067, upper bound: 0.7200039
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7109749, upper bound: 0.7199043
time: 6.56 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 16.28 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7097134, upper bound: 0.7078961
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7104418, upper bound: 0.7071788
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7126032, upper bound: 0.7070260
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7077377, upper bound: 0.7102825
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.6770374, upper bound: 0.6836424
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.6828769, upper bound: 0.6778059
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7011955, upper bound: 0.6988579
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7017192, upper bound: 0.6984311
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7216576, upper bound: 0.7275193
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7251592, upper bound: 0.7240118
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7206198, upper bound: 0.7278001
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7250281, upper bound: 0.7210399
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7219327, upper bound: 0.7218097
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7219262, upper bound: 0.7218160
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7108067, upper bound: 0.7200039
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 16.28
Output dim: 6, lower bound: -0.7109749, upper bound: 0.7199043

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4490194, 1.4519591
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4516311, 1.4541793
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3770576, 1.3562098
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2436624, 1.2532396
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7736454, 1.7735548
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9866047, 0.9890742
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2788358, 1.2783947
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0567777, 1.0550134
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8143337, 0.8136076
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7517095, 1.7518902

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2130

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6995007, upper bound: 0.6956362
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6975681, upper bound: 0.6975365
time: 4.81 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4519644, 1.4485154
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4538026, 1.4527497
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3563828, 1.3666077
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2480121, 1.2435975
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7767906, 1.7740326
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9884245, 0.9871874
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2793450, 1.2790132
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0550468, 1.0567358
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8190227, 0.8134444
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7517810, 1.7521439

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 3102

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2827

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7101551, upper bound: 0.7068712
time: 5.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7101417, upper bound: 0.7068826
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4502788, 1.4509516
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4457340, 1.4465756
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3554111, 1.3505778
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2335844, 1.2334495
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7464628, 1.7531562
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9701653, 0.9799812
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2986217, 1.2901335
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0677030, 1.0685515
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8165596, 0.8123651
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7441821, 1.7463341

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 414

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 75

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 613

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7125531, upper bound: 0.7058789
time: 7.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7114649, upper bound: 0.7069752
time: 4.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4505963, 1.4506340
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4459829, 1.4463267
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3508801, 1.3545594
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2334285, 1.2334542
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7529411, 1.7466774
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9799066, 0.9702396
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2903390, 1.2984161
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0691173, 1.0671370
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8131094, 0.8158152
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7464442, 1.7440710

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2563

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 330

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7065320, upper bound: 0.7083494
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7065223, upper bound: 0.7083566
time: 5.51 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4518385, 1.4516630
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4418726, 1.4425993
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3521791, 1.3456583
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2196531, 1.2052693
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7613115, 1.7603078
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9822414, 0.9864957
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2707453, 1.2761121
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0573425, 1.0568173
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8181310, 0.8179350
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7556896, 1.7522416

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 660

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6687652, upper bound: 0.6752463
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6687929, upper bound: 0.6752470
time: 4.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4517398, 1.4517612
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4417629, 1.4427092
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3459291, 1.3519087
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2053742, 1.2195482
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7600098, 1.7616100
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9861672, 0.9825699
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2766862, 1.2701716
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0571356, 1.0570242
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8187022, 0.8173637
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7522640, 1.7556682

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1847

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6826177, upper bound: 0.6776280
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6826992, upper bound: 0.6775569
time: 4.68 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4524608, 1.4532619
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4540515, 1.4539759
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3668518, 1.3671136
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2478147, 1.2479205
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7776279, 1.7776608
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9894772, 0.9887798
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2806220, 1.2805386
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0559149, 1.0558152
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8189023, 0.8185461
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7524180, 1.7526479

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1406

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1695

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7011365, upper bound: 0.6987131
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7010503, upper bound: 0.6987993
time: 14.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4528117, 1.4523826
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4541798, 1.4548874
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3669376, 1.3665800
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2480249, 1.2481589
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7773628, 1.7777042
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9884512, 0.9890673
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2811141, 1.2806892
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0561323, 1.0571089
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8192575, 0.8181353
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7536559, 1.7523971

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1502

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 738

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7009501, upper bound: 0.6977273
time: 8.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7015628, upper bound: 0.6981139
time: 7.30 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4416652, 1.4410014
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4524999, 1.4523010
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3295941, 1.3311462
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2125511, 1.2131457
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7727680, 1.7760968
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9430771, 0.9512997
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2586536, 1.2596216
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0137801, 1.0133278
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8185472, 0.8217913
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7401776, 1.7407708

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 913

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2461

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7176879, upper bound: 0.7245016
time: 7.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7186376, upper bound: 0.7235536
time: 6.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4413285, 1.4413381
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4530444, 1.4517565
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3299408, 1.3307991
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2138500, 1.2118464
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7731552, 1.7757096
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9456801, 0.9486964
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2589149, 1.2593603
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0161357, 1.0109727
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8210006, 0.8193378
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7399154, 1.7410326

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1506

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2396

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7177552, upper bound: 0.7140763
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7112519, upper bound: 0.7166728
time: 4.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4046593, 1.4131508
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4545889, 1.4525065
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3455133, 1.3484216
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2069087, 1.2084289
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7662706, 1.7597737
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9875612, 0.9833765
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2470942, 1.2470932
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0222263, 1.0213225
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8153915, 0.8161949
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7340870, 1.7317381

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1997

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7172736, upper bound: 0.7233874
time: 4.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7164111, upper bound: 0.7242284
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4128232, 1.4049869
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4538631, 1.4532351
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3490705, 1.3451114
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2074676, 1.2078700
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7612457, 1.7647996
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9882736, 0.9826016
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2466888, 1.2474985
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0222206, 1.0213284
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8155313, 0.8160553
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7339163, 1.7318230

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1406

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2234

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7211662, upper bound: 0.7206592
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7246675, upper bound: 0.7171521
time: 5.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4510560, 1.4508977
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4540977, 1.4533703
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3655968, 1.3650541
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2460527, 1.2462611
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7768793, 1.7757392
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9903872, 0.9894271
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2781792, 1.2787676
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0400949, 1.0427179
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8182420, 0.8190107
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7481747, 1.7489576

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2148

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2917

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7067107, upper bound: 0.7065882
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7067107, upper bound: 0.7065882
time: 5.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4510565, 1.4508972
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4540973, 1.4533710
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3655930, 1.3650579
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2460423, 1.2462711
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7768850, 1.7757339
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9903829, 0.9894315
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2781730, 1.2787733
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0401096, 1.0427027
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8182470, 0.8190056
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7481747, 1.7489586

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1515

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 75

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7171332, upper bound: 0.7144077
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7145180, upper bound: 0.7170712
time: 5.71 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4512572, 1.4498382
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4529667, 1.4518862
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3634181, 1.3689575
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2454076, 1.2450986
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7756577, 1.7752194
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9874291, 0.9856656
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2778177, 1.2781014
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0388365, 1.0397298
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8138759, 0.8174158
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7478533, 1.7485847

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1705

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2461

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7069859, upper bound: 0.7171554
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7079265, upper bound: 0.7162219
time: 5.09 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4493575, 1.4512129
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4539523, 1.4524994
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3648653, 1.3644514
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2461224, 1.2453675
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7766914, 1.7755656
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9904358, 0.9864104
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2782083, 1.2784185
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0373311, 1.0427127
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8182557, 0.8146710
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7480183, 1.7489433

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 192

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1979

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7098905, upper bound: 0.7178378
time: 4.19 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7089091, upper bound: 0.7188193
time: 4.16 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.82 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.6995007, upper bound: 0.6956362
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.6975681, upper bound: 0.6975365
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7101551, upper bound: 0.7068712
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7101417, upper bound: 0.7068826
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7125531, upper bound: 0.7058789
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7114649, upper bound: 0.7069752
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7065320, upper bound: 0.7083494
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7065223, upper bound: 0.7083566
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.6687652, upper bound: 0.6752463
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.6687929, upper bound: 0.6752470
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.6826177, upper bound: 0.6776280
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.6826992, upper bound: 0.6775569
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7011365, upper bound: 0.6987131
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7010503, upper bound: 0.6987993
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7009501, upper bound: 0.6977273
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7015628, upper bound: 0.6981139
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7176879, upper bound: 0.7245016
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7186376, upper bound: 0.7235536
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7177552, upper bound: 0.7140763
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7112519, upper bound: 0.7166728
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7172736, upper bound: 0.7233874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7164111, upper bound: 0.7242284
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7211662, upper bound: 0.7206592
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7246675, upper bound: 0.7171521
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7067107, upper bound: 0.7065882
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7067107, upper bound: 0.7065882
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7171332, upper bound: 0.7144077
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7145180, upper bound: 0.7170712
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7069859, upper bound: 0.7171554
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7079265, upper bound: 0.7162219
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7098905, upper bound: 0.7178378
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.82
Output dim: 6, lower bound: -0.7089091, upper bound: 0.7188193

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4475837, 1.4501290
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3172934, 1.3289165
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2984343, 1.2714214
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1833811, 1.1989717
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7219257, 1.7203336
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9662113, 0.9670382
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2658973, 1.2613373
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8947625, 0.8886354
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8091829, 0.8081686
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7563396, 1.7565422

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2349

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1228

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6987374, upper bound: 0.6949288
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6987933, upper bound: 0.6948728
time: 4.60 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4471893, 1.4505234
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3263690, 1.3198414
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2922692, 1.2775855
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1893945, 1.1929584
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7204247, 1.7218351
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9645686, 0.9686809
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2617784, 1.2654562
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8903999, 0.8929977
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8088949, 0.8084567
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7563615, 1.7565203

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 890

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 613

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6975200, upper bound: 0.6966780
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6967880, upper bound: 0.6974869
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4519644, 1.4485154
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4538021, 1.4527488
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3563790, 1.3666043
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2480111, 1.2435970
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7767897, 1.7740321
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9884245, 0.9871871
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2793336, 1.2789984
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0550387, 1.0567296
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8190134, 0.8134370
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7517781, 1.7521420

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1695

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1705

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7055489, upper bound: 0.7026160
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7055794, upper bound: 0.7026338
time: 4.79 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4519649, 1.4485154
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4538016, 1.4527493
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3563795, 1.3666039
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2480106, 1.2435970
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7767906, 1.7740321
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9884243, 0.9871874
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2793307, 1.2790012
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0550416, 1.0567274
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8190153, 0.8134351
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7517781, 1.7521415

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2461

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7045249, upper bound: 0.7015997
time: 4.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7045249, upper bound: 0.7015897
time: 4.90 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4456115, 1.4464140
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4440713, 1.4447236
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3430510, 1.3385277
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2216506, 1.2218652
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7463980, 1.7531919
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9647987, 0.9733669
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2982669, 1.2897453
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0655336, 1.0649037
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8089231, 0.8069128
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7436256, 1.7458029

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 738

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1695

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7124874, upper bound: 0.7056860
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7123429, upper bound: 0.7058224
time: 4.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4457412, 1.4462843
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4438820, 1.4449127
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3433590, 1.3382177
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2220001, 1.2215157
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7464981, 1.7530909
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9635510, 0.9746144
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2982340, 1.2897782
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0640554, 1.0663829
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8111051, 0.8047286
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7436504, 1.7457795

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1502

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2356

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7078763, upper bound: 0.7068759
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7113520, upper bound: 0.7034013
time: 7.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4503951, 1.4505925
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4457216, 1.4461350
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3503757, 1.3543267
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2332425, 1.2332783
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7529039, 1.7466211
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9798868, 0.9700406
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2903209, 1.2983885
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0685246, 1.0668578
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8130825, 0.8159108
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7463431, 1.7438350

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2818

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1997

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 3, pos: 76

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7037445, upper bound: 0.7047936
time: 7.94 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7029631, upper bound: 0.7055616
time: 6.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4505963, 1.4504328
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4457912, 1.4463267
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3506474, 1.3545594
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2332525, 1.2334542
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7529411, 1.7466397
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9797072, 0.9702396
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2903390, 1.2983980
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0688384, 1.0671370
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8131094, 0.8157883
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7464442, 1.7439704

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1695

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2234

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7025490, upper bound: 0.7080649
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7061668, upper bound: 0.7042095
time: 7.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4484234, 1.4489059
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4420247, 1.4425445
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3396716, 1.3434911
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2047400, 1.2190518
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7593603, 1.7589164
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9798183, 0.9776167
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2766042, 1.2702079
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0556326, 1.0554676
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8180583, 0.8166240
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7404003, 1.7446117

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1775

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 3102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6788515, upper bound: 0.6753975
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6805038, upper bound: 0.6732980
time: 4.45 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4488859, 1.4484439
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4415984, 1.4429708
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3375125, 1.3456497
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2048769, 1.2189155
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7573147, 1.7609620
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9812136, 0.9762214
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2767224, 1.2700896
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0555782, 1.0555212
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8179624, 0.8167198
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7412081, 1.7438035

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2827

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6824126, upper bound: 0.6768660
time: 4.53 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6824083, upper bound: 0.6772999
time: 6.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4501791, 1.4506717
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4518533, 1.4510648
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3674464, 1.3679028
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2473731, 1.2474675
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7801409, 1.7799916
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9878287, 0.9871109
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2783914, 1.2780824
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0377417, 1.0413883
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8186021, 0.8162951
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7513485, 1.7518654

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 2563

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2130

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6692014, upper bound: 0.6717002
time: 5.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6748304, upper bound: 0.6664357
time: 4.53 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4498706, 1.4509802
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4511404, 1.4517777
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3676405, 1.3677087
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2473617, 1.2474790
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7799578, 1.7801747
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9878080, 0.9871315
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2781672, 1.2783065
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0414877, 1.0376418
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8166511, 0.8182460
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7516356, 1.7515779

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 75

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1515

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6967203, upper bound: 0.6984776
time: 7.39 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7007087, upper bound: 0.6942139
time: 4.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4527192, 1.4522972
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4542079, 1.4549022
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3668694, 1.3665290
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2479086, 1.2480569
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7773600, 1.7777014
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9884667, 0.9890864
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2811961, 1.2807813
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0561929, 1.0571599
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8192890, 0.8181540
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7536941, 1.7524247

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2858

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6966671, upper bound: 0.6927099
time: 5.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6943433, upper bound: 0.6943948
time: 4.12 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4527254, 1.4522910
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4541945, 1.4549153
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3668871, 1.3665113
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2479243, 1.2480416
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7773600, 1.7777019
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9884701, 0.9890831
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2812057, 1.2807717
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0561833, 1.0571690
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8192763, 0.8181665
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7536836, 1.7524343

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 1997

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 1479

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6953321, upper bound: 0.6894029
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6921755, upper bound: 0.6915400
time: 4.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4412556, 1.4409037
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4521646, 1.4530406
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3297901, 1.3311462
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2125506, 1.2127113
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7708817, 1.7756476
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9429336, 0.9512997
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2583108, 1.2595401
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0136971, 1.0129807
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8185472, 0.8222744
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7395720, 1.7382312

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 2305
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 2901

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7077999, upper bound: 0.7127708
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7061302, upper bound: 0.7145917
time: 5.15 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 14.50 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6987374, upper bound: 0.6949288
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6987933, upper bound: 0.6948728
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6975200, upper bound: 0.6966780
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6967880, upper bound: 0.6974869
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7055489, upper bound: 0.7026160
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7055794, upper bound: 0.7026338
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7045249, upper bound: 0.7015997
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7045249, upper bound: 0.7015897
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7124874, upper bound: 0.7056860
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7123429, upper bound: 0.7058224
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7078763, upper bound: 0.7068759
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7113520, upper bound: 0.7034013
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7037445, upper bound: 0.7047936
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7029631, upper bound: 0.7055616
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7025490, upper bound: 0.7080649
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7061668, upper bound: 0.7042095
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6788515, upper bound: 0.6753975
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6805038, upper bound: 0.6732980
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6824126, upper bound: 0.6768660
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6824083, upper bound: 0.6772999
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6692014, upper bound: 0.6717002
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6748304, upper bound: 0.6664357
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6967203, upper bound: 0.6984776
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7007087, upper bound: 0.6942139
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6966671, upper bound: 0.6927099
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6943433, upper bound: 0.6943948
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6953321, upper bound: 0.6894029
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.6921755, upper bound: 0.6915400
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7077999, upper bound: 0.7127708
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.50
Output dim: 6, lower bound: -0.7061302, upper bound: 0.7145917
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7186376, upper bound: 0.7235536
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7177552, upper bound: 0.7140763
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7112519, upper bound: 0.7166728
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7172736, upper bound: 0.7233874
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7164111, upper bound: 0.7242284
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7211662, upper bound: 0.7206592
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7246675, upper bound: 0.7171521
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7067107, upper bound: 0.7065882
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7067107, upper bound: 0.7065882
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7171332, upper bound: 0.7144077
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7145180, upper bound: 0.7170712
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7069859, upper bound: 0.7171554
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7079265, upper bound: 0.7162219
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7098905, upper bound: 0.7178378
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.50
Output dim: 6, lower bound: -0.7089091, upper bound: 0.7188193
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.2817935943603516
rel_dist={6: [-0.7326686905779951, 0.7326686039477615]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2411.72 seconds
