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
execution time: IAR + LP analysis = 15.13 + 32.05 = 47.17 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.83 seconds, max iter: 100)

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
Binary search time: 208.58 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Relational Split (RS_dual_Z) starts
Time budget: 3344.25 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: RSZ, layer: 3, pos: 2130

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0936794
time: 3.39 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0936775, upper bound: 1.0884660
time: 3.32 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.02
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0936794
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.02
Output dim: 6, lower bound: -1.0936775, upper bound: 1.0884660

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6832390, 1.6830091
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9554915, 1.9552350
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7655573, 1.7509737
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6039605, 1.5706425
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3117673, 1.3209276
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6041493, 1.6180115
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4761086, 1.4756255
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1293132, 1.1306461
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1529560, 2.1449609

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0771431, upper bound: 1.0826762
time: 3.98 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0771457, upper bound: 1.0826664
time: 3.99 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6830091, 1.6840801
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9677978, 1.9554915
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7509737, 1.7719808
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5706425, 1.6135306
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3182647, 1.3117676
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6146688, 1.6041496
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4756255, 1.4759169
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1304395, 1.1293131
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1449614, 2.1463537

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0826668, upper bound: 1.0771458
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0826743, upper bound: 1.0771433
time: 3.79 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.12 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.12
Output dim: 6, lower bound: -1.0771431, upper bound: 1.0826762
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.12
Output dim: 6, lower bound: -1.0771457, upper bound: 1.0826664
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.12
Output dim: 6, lower bound: -1.0826668, upper bound: 1.0771458
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.12
Output dim: 6, lower bound: -1.0826743, upper bound: 1.0771433

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6862950, 1.6816330
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9545064, 1.9528193
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7641096, 1.7600422
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6032453, 1.5692992
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3087604, 1.3161829
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6037593, 1.6168811
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4766376, 1.4726427
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1249332, 1.1326709
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1522107, 2.1446033

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0640925, upper bound: 1.0634175
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0575111, upper bound: 1.0701918
time: 4.05 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6818628, 1.6830091
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9554915, 1.9542499
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7655573, 1.7495270
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.6039605, 1.5699272
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3117673, 1.3179207
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6041493, 1.6176214
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4731252, 1.4756255
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1293132, 1.1262662
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1525979, 2.1449609

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0651190, upper bound: 1.0619560
time: 4.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0590583, upper bound: 1.0691705
time: 3.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6860657, 1.6827044
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9668126, 1.9530759
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7495270, 1.7810493
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5699272, 1.6121879
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3152578, 1.3070228
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6142774, 1.6030190
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4761550, 1.4729340
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1260602, 1.1313378
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1442161, 2.1459951

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0691685, upper bound: 1.0590585
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0619542, upper bound: 1.0651194
time: 3.73 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6816330, 1.6840801
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.9677978, 1.9545064
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.7509737, 1.7705345
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5706425, 1.6128159
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3182647, 1.3087606
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6146688, 1.6037595
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.4726427, 1.4759169
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1304395, 1.1249331
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1446033, 2.1463537

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0701898, upper bound: 1.0575114
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0634156, upper bound: 1.0640926
time: 3.79 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.08 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 6, lower bound: -1.0640925, upper bound: 1.0634175
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 6, lower bound: -1.0575111, upper bound: 1.0701918
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 6, lower bound: -1.0651190, upper bound: 1.0619560
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 6, lower bound: -1.0590583, upper bound: 1.0691705
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 6, lower bound: -1.0691685, upper bound: 1.0590585
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 6, lower bound: -1.0619542, upper bound: 1.0651194
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 6, lower bound: -1.0701898, upper bound: 1.0575114
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.08
Output dim: 6, lower bound: -1.0634156, upper bound: 1.0640926

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6854658, 1.6798239
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8206325, 1.8413312
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6886744, 1.6744492
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5427685, 1.5238233
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2881413, 1.2967441
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5910454, 1.5946105
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3202062, 1.3060205
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1196856, 1.1281257
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1573644, 2.1493907

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0424012, upper bound: 1.0413009
time: 3.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0424012, upper bound: 1.0413009
time: 3.84 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6844854, 1.6807437
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8418088, 1.8189456
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6785178, 1.6787994
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5589991, 1.5088224
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2843080, 1.2955635
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5814342, 1.6041670
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3100157, 1.3161998
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1190143, 1.1274235
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1569982, 2.1493397

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0352085, upper bound: 1.0481476
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0352085, upper bound: 1.0481479
time: 4.10 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6809735, 1.6811996
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8208995, 1.8397553
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6900725, 1.6639345
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5434828, 1.5234816
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2911611, 1.2924376
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5914364, 1.5952954
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3165803, 1.3085369
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1241632, 1.1203473
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1573339, 2.1495676

Time for backsubstitution: 5.42 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0435025, upper bound: 1.0394889
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0435025, upper bound: 1.0394889
time: 3.83 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6800532, 1.6821194
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8420758, 1.8203762
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6799150, 1.6783173
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5597138, 1.5094504
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2873282, 1.2973011
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5818253, 1.6049073
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3065033, 1.3187163
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1234915, 1.1210188
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1573853, 2.1495161

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0370189, upper bound: 1.0470462
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0370189, upper bound: 1.0470462
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6852360, 1.6808748
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8331928, 1.8433847
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6783175, 1.6963103
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5094504, 1.5669076
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2948518, 1.2886145
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6013393, 1.5807495
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3198256, 1.3063114
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1208127, 1.1267924
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1493707, 2.1508274

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0470458, upper bound: 1.0370190
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0470458, upper bound: 1.0370190
time: 3.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6842561, 1.6817946
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8543692, 1.8192022
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6639342, 1.7006605
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5234818, 1.5519068
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2910190, 1.2864034
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5917287, 1.5903049
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3095326, 1.3164907
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1201410, 1.1260904
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1490035, 2.1507764

Time for backsubstitution: 5.41 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0394889, upper bound: 1.0435026
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0394889, upper bound: 1.0435026
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6807437, 1.6822491
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8334594, 1.8418090
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6797156, 1.6857955
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5101652, 1.5665660
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2978721, 1.2843082
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6017299, 1.5814345
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3161998, 1.3088282
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1252894, 1.1190141
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1493392, 2.1510043

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0481475, upper bound: 1.0352089
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0481475, upper bound: 1.0352089
time: 3.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6798239, 1.6831694
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8546357, 1.8206327
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6653323, 1.7001784
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5241966, 1.5525348
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2940392, 1.2881410
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5921192, 1.5910454
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3060207, 1.3190076
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1246176, 1.1196858
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1493907, 2.1509533

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0413008, upper bound: 1.0424013
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0413008, upper bound: 1.0424013
time: 3.70 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 13.02 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0424012, upper bound: 1.0413009
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0424012, upper bound: 1.0413009
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0352085, upper bound: 1.0481476
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0352085, upper bound: 1.0481479
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0435025, upper bound: 1.0394889
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0435025, upper bound: 1.0394889
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0370189, upper bound: 1.0470462
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0370189, upper bound: 1.0470462
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0470458, upper bound: 1.0370190
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0470458, upper bound: 1.0370190
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0394889, upper bound: 1.0435026
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0394889, upper bound: 1.0435026
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0481475, upper bound: 1.0352089
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0481475, upper bound: 1.0352089
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0413008, upper bound: 1.0424013
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 13.02
Output dim: 6, lower bound: -1.0413008, upper bound: 1.0424013

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6851821, 1.6789050
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8175793, 1.8355198
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6839929, 1.6697049
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5386147, 1.5216720
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2869506, 1.2877142
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5709648, 1.5802503
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3131123, 1.2943366
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1041420, 1.1352744
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1543651, 2.1328683

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0387271, upper bound: 1.0341134
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0338568, upper bound: 1.0380581
time: 4.16 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6854658, 1.6795402
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8148212, 1.8413312
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6886744, 1.6697676
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5427685, 1.5196698
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2791114, 1.2967441
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5766850, 1.5946105
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3202062, 1.2989265
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1196856, 1.1125818
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1408420, 2.1493907

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0387271, upper bound: 1.0341134
time: 3.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0338546, upper bound: 1.0380567
time: 4.30 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6842022, 1.6798248
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8387585, 1.8131342
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6738362, 1.6740561
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5548458, 1.5066714
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2831192, 1.2865336
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5613546, 1.5898066
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3029218, 1.3045168
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1034703, 1.1345723
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1539998, 2.1328168

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0319182, upper bound: 1.0394216
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0282895, upper bound: 1.0444900
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6844854, 1.6804600
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8359976, 1.8189456
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6785178, 1.6741176
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5589991, 1.5046690
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2752786, 1.2955635
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5670738, 1.6041670
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3100157, 1.3091059
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1190143, 1.1118797
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1404757, 2.1493397

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0319182, upper bound: 1.0394216
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0282832, upper bound: 1.0444903
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6806898, 1.6802807
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8178458, 1.8339441
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6853900, 1.6593425
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5393362, 1.5213304
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2899704, 1.2834077
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5713558, 1.5809350
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3094864, 1.2968529
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1084700, 1.1280873
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1543345, 2.1330309

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0398625, upper bound: 1.0324800
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0347936, upper bound: 1.0361880
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6809735, 1.6809158
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8150878, 1.8397553
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6900725, 1.6592529
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5434828, 1.5193281
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2821312, 1.2924376
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5770760, 1.5952954
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3165803, 1.3014429
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1241632, 1.1048036
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1408114, 2.1495676

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0398625, upper bound: 1.0324860
time: 3.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0347936, upper bound: 1.0361880
time: 3.87 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6797695, 1.6812005
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8390250, 1.8145647
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6752334, 1.6737261
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5555668, 1.5072994
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2861390, 1.2882712
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5617456, 1.5905471
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.2994094, 1.3070332
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1077981, 1.1287588
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1543860, 2.1329794

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0338020, upper bound: 1.0384777
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0299193, upper bound: 1.0433549
time: 3.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6800532, 1.6818357
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8362641, 1.8203762
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6799150, 1.6736357
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5597138, 1.5052969
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2782984, 1.2973011
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5674648, 1.6049073
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3065033, 1.3116221
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1234915, 1.1054749
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1408629, 2.1495161

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0338020, upper bound: 1.0384818
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0299193, upper bound: 1.0433549
time: 3.87 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6849523, 1.6799560
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8301392, 1.8375733
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6736360, 1.6915660
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5052967, 1.5647564
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2936611, 1.2795846
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5812588, 1.5663893
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3127317, 1.2946275
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1052685, 1.1339412
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1463714, 2.1343045

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0433548, upper bound: 1.0299194
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0384814, upper bound: 1.0338021
time: 3.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6852360, 1.6805911
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8273811, 1.8433847
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6783175, 1.6916287
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5094504, 1.5627542
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2858219, 1.2886145
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5869789, 1.5807495
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3198256, 1.2992175
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1208127, 1.1112487
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1328483, 2.1508274

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0433548, upper bound: 1.0299194
time: 3.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0384773, upper bound: 1.0338020
time: 4.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6839724, 1.6808758
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8513184, 1.8133907
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6592526, 1.6959171
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5193281, 1.5497558
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2898293, 1.2773736
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5716491, 1.5759444
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3024387, 1.3048078
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1045969, 1.1332393
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1460052, 2.1342535

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0361879, upper bound: 1.0347939
time: 3.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0324858, upper bound: 1.0398632
time: 4.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6842561, 1.6815109
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8485575, 1.8192022
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6639342, 1.6959786
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5234818, 1.5477533
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2819891, 1.2864034
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5773683, 1.5903049
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3095326, 1.3093967
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1201410, 1.1105466
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1324811, 2.1507764

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0361879, upper bound: 1.0347939
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0324798, upper bound: 1.0398626
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6804600, 1.6813307
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8304062, 1.8359973
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6750331, 1.6812036
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5060182, 1.5644150
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2966814, 1.2752783
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5816493, 1.5670741
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3091059, 1.2971443
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1095963, 1.1267540
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1463408, 2.1344671

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0444903, upper bound: 1.0282834
time: 3.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0394216, upper bound: 1.0319179
time: 3.89 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6807437, 1.6819658
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8276482, 1.8418090
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6797156, 1.6811140
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5101652, 1.5624127
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2888422, 1.2843082
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5873694, 1.5814345
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3161998, 1.3017342
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1252894, 1.1034703
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1328177, 2.1510043

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0444903, upper bound: 1.0282897
time: 3.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0394216, upper bound: 1.0319179
time: 3.87 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6795402, 1.6822505
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8515854, 1.8148212
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6606498, 1.6955872
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5200496, 1.5503838
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2928495, 1.2791111
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5720396, 1.5766850
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.2989264, 1.3073245
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1089246, 1.1274257
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1463923, 2.1344156

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0380566, upper bound: 1.0338546
time: 3.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0341133, upper bound: 1.0387273
time: 4.03 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6798239, 1.6828856
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8488245, 1.8206327
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6653323, 1.6954968
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5241966, 1.5483813
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2850089, 1.2881410
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5777588, 1.5910454
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3060207, 1.3119135
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1246176, 1.1041420
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1328683, 2.1509533

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0380566, upper bound: 1.0338568
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0341133, upper bound: 1.0387274
time: 3.96 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.43 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0387271, upper bound: 1.0341134
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0338568, upper bound: 1.0380581
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0387271, upper bound: 1.0341134
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0338546, upper bound: 1.0380567
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0319182, upper bound: 1.0394216
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0282895, upper bound: 1.0444900
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0319182, upper bound: 1.0394216
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0282832, upper bound: 1.0444903
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0398625, upper bound: 1.0324800
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0347936, upper bound: 1.0361880
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0398625, upper bound: 1.0324860
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0347936, upper bound: 1.0361880
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0338020, upper bound: 1.0384777
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0299193, upper bound: 1.0433549
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0338020, upper bound: 1.0384818
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0299193, upper bound: 1.0433549
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0433548, upper bound: 1.0299194
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0384814, upper bound: 1.0338021
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0433548, upper bound: 1.0299194
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0384773, upper bound: 1.0338020
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0361879, upper bound: 1.0347939
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0324858, upper bound: 1.0398632
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0361879, upper bound: 1.0347939
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0324798, upper bound: 1.0398626
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0444903, upper bound: 1.0282834
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0394216, upper bound: 1.0319179
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0444903, upper bound: 1.0282897
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0394216, upper bound: 1.0319179
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0380566, upper bound: 1.0338546
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0341133, upper bound: 1.0387273
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0380566, upper bound: 1.0338568
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.43
Output dim: 6, lower bound: -1.0341133, upper bound: 1.0387274

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6831393, 1.6776032
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8235598, 1.8477113
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6856604, 1.6565030
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5300102, 1.5184317
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2766392, 1.3087416
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5988617, 1.5880001
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3127108, 1.2965374
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1011537, 1.1242356
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1495190, 2.1304512

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0259384, upper bound: 1.0222082
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0267700, upper bound: 1.0216380
time: 3.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6835313, 1.6768622
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8235378, 1.8415008
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6707907, 1.6626594
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5299988, 1.5130677
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2993705, 1.2774031
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5787148, 1.6073256
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3131456, 1.2939355
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.0931033, 1.1293354
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1519480, 2.1251702

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0210680, upper bound: 1.0261482
time: 4.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0219014, upper bound: 1.0255802
time: 4.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6834230, 1.6782384
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8208022, 1.8535242
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6903505, 1.6565657
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5341754, 1.5164273
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2688000, 1.3163373
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6037617, 1.6021419
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3198528, 1.3018260
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1170195, 1.1015432
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1359959, 2.1462903

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0259384, upper bound: 1.0222082
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0267700, upper bound: 1.0216380
time: 3.80 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6838150, 1.6774974
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8213825, 1.8473139
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6754799, 1.6626427
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5341640, 1.5110652
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2915299, 1.2849987
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5844350, 1.6214671
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3202868, 1.2985253
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1089691, 1.1061736
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1384249, 2.1410098

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0210658, upper bound: 1.0261500
time: 3.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0218991, upper bound: 1.0255820
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6821594, 1.6783276
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8447390, 1.8208523
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6712065, 1.6608541
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5462408, 1.4990273
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2728078, 1.3073115
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5892520, 1.5975564
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3025208, 1.3035467
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1004838, 1.1235336
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1460438, 2.1304002

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0190778, upper bound: 1.0275136
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0199627, upper bound: 1.0269071
time: 4.05 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6825523, 1.6777825
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8504004, 1.8191152
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6606340, 1.6712203
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5516024, 1.4980669
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3013666, 1.2762222
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5691047, 1.6168845
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3029547, 1.3041158
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.0924317, 1.1284617
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1515827, 2.1278529

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0154245, upper bound: 1.0325824
time: 4.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0163456, upper bound: 1.0319773
time: 4.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6824431, 1.6789627
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8419781, 1.8266652
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6758957, 1.6609156
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5504060, 1.4970226
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2649667, 1.3149073
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5941520, 1.6116982
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3096619, 1.3088346
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1163496, 1.1008410
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1325197, 2.1462393

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0190778, upper bound: 1.0275136
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0199627, upper bound: 1.0269071
time: 4.13 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6828361, 1.6784177
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8482356, 1.8249283
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6653233, 1.6712034
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5557675, 1.4960644
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2935245, 1.2838180
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5748239, 1.6310263
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3100967, 1.3087047
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1082972, 1.1052998
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1380596, 2.1436925

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0154184, upper bound: 1.0325824
time: 4.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0163396, upper bound: 1.0319772
time: 4.28 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6786470, 1.6795225
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8238535, 1.8461823
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6883631, 1.6461406
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5305490, 1.5180893
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2786782, 1.3016541
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5992527, 1.5886850
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3090849, 1.2997637
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1059674, 1.1170486
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1491127, 2.1306119

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0270738, upper bound: 1.0205720
time: 3.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0279054, upper bound: 1.0200040
time: 3.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6791921, 1.6787820
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8238316, 1.8399248
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6734924, 1.6569908
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5305376, 1.5127261
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3014095, 1.2730963
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5791059, 1.6080127
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3092155, 1.2971618
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.0979167, 1.1242840
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1519184, 2.1253314

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0220049, upper bound: 1.0242815
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0228364, upper bound: 1.0237104
time: 4.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6789308, 1.6801577
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8210959, 1.8519955
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6930513, 1.6460507
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5347137, 1.5160849
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2708390, 1.3092501
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6041527, 1.6028266
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3162270, 1.3050523
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1218330, 1.0937648
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1358471, 2.1464677

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0270738, upper bound: 1.0205781
time: 3.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0279054, upper bound: 1.0200101
time: 3.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6794758, 1.6794171
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8216763, 1.8457379
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6781816, 1.6566231
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5347018, 1.5107236
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2935688, 1.2806923
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5848260, 1.6221542
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3163567, 1.3017516
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1137823, 1.1018170
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1383934, 2.1411872

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0220049, upper bound: 1.0242812
time: 4.34 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0228364, upper bound: 1.0237104
time: 4.11 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6777272, 1.6802468
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8450332, 1.8211261
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6739082, 1.6605241
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5467796, 1.4986856
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2748468, 1.3006899
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5896430, 1.5982969
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.2990084, 1.3067729
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1052972, 1.1177201
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1464310, 2.1305609

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0209633, upper bound: 1.0265716
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0218459, upper bound: 1.0259642
time: 3.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6784682, 1.6797018
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8506947, 1.8205457
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6633358, 1.6756718
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5521407, 1.4986949
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3034055, 1.2779601
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5694957, 1.6176238
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3023090, 1.3073421
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.0972451, 1.1249565
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1519699, 2.1280141

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0170539, upper bound: 1.0314451
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0179768, upper bound: 1.0308421
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6780109, 1.6808820
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8422723, 1.8269391
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6785975, 1.6604335
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5509443, 1.4966810
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2670057, 1.3082855
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5945425, 1.6124387
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3061495, 1.3120608
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1211628, 1.0944363
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1331654, 2.1464167

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0209633, upper bound: 1.0265738
time: 4.00 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0218459, upper bound: 1.0259674
time: 3.95 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6787515, 1.6803370
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8485298, 1.8263586
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6680250, 1.6753039
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5563054, 1.4966924
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2955635, 1.2855558
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5752149, 1.6317654
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3094501, 1.3119310
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1131107, 1.1024868
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1384459, 2.1438699

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0170539, upper bound: 1.0314451
time: 4.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0179768, upper bound: 1.0308421
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6829100, 1.6786532
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8361197, 1.8497648
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6753035, 1.6783640
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.4966927, 1.5615158
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2833502, 1.3006120
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6091561, 1.5741391
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3123302, 1.2968285
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1022805, 1.1229025
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1415253, 2.1318870

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0308403, upper bound: 1.0179768
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0314450, upper bound: 1.0170540
time: 3.63 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6833014, 1.6779122
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8360972, 1.8435543
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6604338, 1.6845205
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.4966812, 1.5561516
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3060815, 1.2692735
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5890088, 1.5934646
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3127651, 1.2942266
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.0942296, 1.1280022
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1439533, 2.1266065

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0259674, upper bound: 1.0218459
time: 4.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0265736, upper bound: 1.0209633
time: 4.25 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6831937, 1.6792884
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8333621, 1.8555777
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6799936, 1.6784270
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5008574, 1.5595112
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2755110, 1.3082078
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6140561, 1.5882807
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3194723, 1.3021173
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1181458, 1.1002100
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1280012, 2.1477270

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0308403, upper bound: 1.0179768
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0314450, upper bound: 1.0170540
time: 3.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6835852, 1.6785474
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8339419, 1.8493671
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6651230, 1.6845040
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5008459, 1.5541492
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2982409, 1.2768693
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5947289, 1.6076064
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3199062, 1.2988167
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1100951, 1.1048404
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1304302, 2.1424465

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0259641, upper bound: 1.0218459
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0265713, upper bound: 1.0209651
time: 4.17 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6819301, 1.6793771
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8572993, 1.8211087
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6566229, 1.6827152
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5107236, 1.5421114
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2795188, 1.2981515
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5995464, 1.5836945
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3020372, 1.3038380
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1016103, 1.1222006
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1380491, 2.1318359

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0237103, upper bound: 1.0228364
time: 3.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0242797, upper bound: 1.0220056
time: 9.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6823230, 1.6788325
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8629603, 1.8193717
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6460505, 1.6930814
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5160847, 1.5411508
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3080776, 1.2670622
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5793991, 1.6030223
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3024721, 1.3044069
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.0935582, 1.1271286
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1435881, 2.1292892

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0200098, upper bound: 1.0279072
time: 4.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0205780, upper bound: 1.0270739
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6822133, 1.6800122
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8545384, 1.8269217
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6613131, 1.6827767
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5148888, 1.5401068
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2716777, 1.3057473
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6044459, 1.5978360
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3091793, 1.3091259
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1174756, 1.0995079
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1245251, 2.1476755

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0237103, upper bound: 1.0228364
time: 3.95 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0242797, upper bound: 1.0220052
time: 4.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6826062, 1.6794677
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8607955, 1.8251846
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6507406, 1.6930645
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.5202498, 1.5391483
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.3002355, 1.2746580
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.5851183, 1.6171641
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3096132, 1.3089960
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1094235, 1.1039667
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1300640, 2.1451292

Time for backsubstitution: 5.50 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0200038, upper bound: 1.0279054
time: 4.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0205720, upper bound: 1.0270756
time: 4.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.6784177, 1.6805725
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.8364143, 1.8482358
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.6780062, 1.6680019
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.4972310, 1.5611734
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.9205575, 1.9205575
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.2853887, 1.2935250
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.6095476, 1.5748239
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.3087044, 1.3000546
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -1.1070937, 1.1157154
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -2.1411180, 2.1320491

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0319758, upper bound: 1.0163398
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0325806, upper bound: 1.0154184
time: 3.77 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 13.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0259384, upper bound: 1.0222082
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0267700, upper bound: 1.0216380
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0210680, upper bound: 1.0261482
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0219014, upper bound: 1.0255802
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0259384, upper bound: 1.0222082
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0267700, upper bound: 1.0216380
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0210658, upper bound: 1.0261500
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0218991, upper bound: 1.0255820
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0190778, upper bound: 1.0275136
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0199627, upper bound: 1.0269071
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0154245, upper bound: 1.0325824
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0163456, upper bound: 1.0319773
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0190778, upper bound: 1.0275136
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0199627, upper bound: 1.0269071
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0154184, upper bound: 1.0325824
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0163396, upper bound: 1.0319772
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0270738, upper bound: 1.0205720
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0279054, upper bound: 1.0200040
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0220049, upper bound: 1.0242815
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0228364, upper bound: 1.0237104
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0270738, upper bound: 1.0205781
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0279054, upper bound: 1.0200101
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0220049, upper bound: 1.0242812
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0228364, upper bound: 1.0237104
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0209633, upper bound: 1.0265716
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0218459, upper bound: 1.0259642
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0170539, upper bound: 1.0314451
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0179768, upper bound: 1.0308421
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0209633, upper bound: 1.0265738
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0218459, upper bound: 1.0259674
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0170539, upper bound: 1.0314451
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0179768, upper bound: 1.0308421
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0308403, upper bound: 1.0179768
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0314450, upper bound: 1.0170540
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0259674, upper bound: 1.0218459
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0265736, upper bound: 1.0209633
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0308403, upper bound: 1.0179768
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0314450, upper bound: 1.0170540
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0259641, upper bound: 1.0218459
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0265713, upper bound: 1.0209651
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0237103, upper bound: 1.0228364
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0242797, upper bound: 1.0220056
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0200098, upper bound: 1.0279072
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0205780, upper bound: 1.0270739
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0237103, upper bound: 1.0228364
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0242797, upper bound: 1.0220052
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0200038, upper bound: 1.0279054
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0205720, upper bound: 1.0270756
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0319758, upper bound: 1.0163398
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 13.22
Output dim: 6, lower bound: -1.0325806, upper bound: 1.0154184
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 6, lower bound: -1.0394216, upper bound: 1.0319179
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 6, lower bound: -1.0444903, upper bound: 1.0282897
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 6, lower bound: -1.0394216, upper bound: 1.0319179
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 6, lower bound: -1.0380566, upper bound: 1.0338546
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 6, lower bound: -1.0341133, upper bound: 1.0387273
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 6, lower bound: -1.0380566, upper bound: 1.0338568
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.22
Output dim: 6, lower bound: -1.0341133, upper bound: 1.0387274
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.614668846130371
rel_dist={6: [-1.1326113226070742, 1.1326113586240112]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2130

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8153173
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8153156, upper bound: 0.8089548
time: 3.82 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 7.80 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 7.80
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8153173
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 7.80
Output dim: 6, lower bound: -0.8153156, upper bound: 0.8089548

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5082512, 1.5081201
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5740442, 1.5738974
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4583893, 1.4500556
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3157654, 1.2967267
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8415699, 1.8398337
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0652068, 1.0704409
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3544927, 1.3624139
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1625123, 1.1622362
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8962016, 0.8969634
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8575792, 1.8530111

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8031658, upper bound: 0.8096184
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8032987, upper bound: 0.8094833
time: 4.54 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5081201, 1.5091910
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5863504, 1.5740440
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4500561, 1.4710631
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2967267, 1.3396149
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8576651, 1.8415704
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0717037, 1.0652066
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3650122, 1.3544927
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1622362, 1.1625276
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8973279, 0.8962016
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8530111, 1.8544040

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8094828, upper bound: 0.8032988
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8096180, upper bound: 0.8031656
time: 4.36 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.43 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.43
Output dim: 6, lower bound: -0.8031658, upper bound: 0.8096184
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.43
Output dim: 6, lower bound: -0.8032987, upper bound: 0.8094833
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.43
Output dim: 6, lower bound: -0.8094828, upper bound: 0.8032988
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.43
Output dim: 6, lower bound: -0.8096180, upper bound: 0.8031656

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5094080, 1.5067439
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5730586, 1.5720949
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4569426, 1.4546175
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3150501, 1.2956524
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8398457, 1.8394876
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0621998, 1.0664409
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3541026, 1.3616009
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1615360, 1.1592534
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8918216, 0.8962432
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8570008, 1.8526535

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7912114, upper bound: 0.7948039
time: 8.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7875484, upper bound: 0.7981896
time: 4.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5068750, 1.5081201
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5740442, 1.5729122
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4583893, 1.4486094
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.3157654, 1.2960114
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8412247, 1.8398337
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0652068, 1.0674340
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3544927, 1.3620238
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1595294, 1.1622362
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8962016, 0.8925834
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8572221, 1.8530111

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7920217, upper bound: 0.7931054
time: 4.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7893919, upper bound: 0.7974052
time: 7.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5092769, 1.5078154
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5853648, 1.5722413
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4486094, 1.4756250
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2960114, 1.3385410
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8559399, 1.8412242
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0686970, 1.0612066
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3646207, 1.3536797
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1612604, 1.1595447
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8929486, 0.8954813
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8524327, 1.8540454

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7974051, upper bound: 0.7893936
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7931055, upper bound: 0.7920214
time: 4.63 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5067439, 1.5091910
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.5863504, 1.5730588
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.4500561, 1.4696164
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2967267, 1.3389001
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8573189, 1.8415704
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0717037, 1.0621996
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3650122, 1.3541026
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.1592529, 1.1625276
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8973279, 0.8918216
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8526540, 1.8544040

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7981898, upper bound: 0.7875484
time: 4.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7948038, upper bound: 0.7912111
time: 5.07 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.98 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 6, lower bound: -0.7912114, upper bound: 0.7948039
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 6, lower bound: -0.7875484, upper bound: 0.7981896
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 6, lower bound: -0.7920217, upper bound: 0.7931054
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 6, lower bound: -0.7893919, upper bound: 0.7974052
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 6, lower bound: -0.7974051, upper bound: 0.7893936
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 6, lower bound: -0.7931055, upper bound: 0.7920214
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 6, lower bound: -0.7981898, upper bound: 0.7875484
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.98
Output dim: 6, lower bound: -0.7948038, upper bound: 0.7912111

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5081587, 1.5049348
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4391856, 1.4510131
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3771539, 1.3690252
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2545733, 1.2437477
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7881260, 1.7834778
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0415802, 1.0464962
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3413887, 1.3434258
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0007372, 0.9926312
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8865743, 0.8913970
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8619971, 1.8574409

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7775826, upper bound: 0.7809890
time: 4.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7775826, upper bound: 0.7809889
time: 5.27 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5075989, 1.5054603
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4512858, 1.4382212
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3713498, 1.3715105
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2638478, 1.2351756
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7860126, 1.7877688
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0393898, 1.0458214
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3358965, 1.3488865
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9949141, 0.9984479
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8861904, 0.8909957
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8617883, 1.8574119

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7735373, upper bound: 0.7845694
time: 6.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7735373, upper bound: 0.7845694
time: 7.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5055914, 1.5063105
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4394522, 1.4501123
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3785510, 1.3630161
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2552881, 1.2435524
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7895050, 1.7838240
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0446000, 1.0440353
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3417797, 1.3438172
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9986653, 0.9951475
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8910515, 0.8869523
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8619800, 1.8576179

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7785245, upper bound: 0.7789644
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7785245, upper bound: 0.7789644
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5050659, 1.5068364
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4515529, 1.4390385
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3727479, 1.3712349
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2645626, 1.2355347
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7875032, 1.7881145
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0424099, 1.0468144
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3362875, 1.3493094
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9929070, 1.0009643
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8906679, 0.8873360
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8620095, 1.8575888

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7756731, upper bound: 0.7836047
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7756731, upper bound: 0.7836047
time: 6.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5080276, 1.5059853
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4517455, 1.4521863
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3712354, 1.3908863
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2355347, 1.2868319
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8042212, 1.7875032
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0482905, 1.0418508
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3516827, 1.3355055
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0005198, 0.9929221
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8877010, 0.8906353
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8574300, 1.8588777

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7836048, upper bound: 0.7756754
time: 4.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7836048, upper bound: 0.7756754
time: 3.98 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5074673, 1.5065112
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4638457, 1.4383676
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3630166, 1.3933716
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2435527, 1.2782602
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8021069, 1.7895050
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0461004, 1.0405872
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3461909, 1.3409657
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9946384, 0.9987388
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8873174, 0.8902340
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8572202, 1.8588486

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7789646, upper bound: 0.7785266
time: 8.63 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7789646, upper bound: 0.7785266
time: 7.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5054603, 1.5073605
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4520125, 1.4512858
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3726335, 1.3848772
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2362490, 1.2866368
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8056002, 1.7878489
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0513108, 1.0393900
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3520732, 1.3358970
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9984479, 0.9954388
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8921778, 0.8861904
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8574119, 1.8590546

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7845701, upper bound: 0.7735373
time: 4.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7845701, upper bound: 0.7735373
time: 4.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5049348, 1.5078859
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4641128, 1.4391851
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3644147, 1.3930960
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2442670, 1.2786188
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8035975, 1.7898512
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0491207, 1.0415802
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3465815, 1.3413887
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9926310, 1.0012556
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8917940, 0.8865743
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8574414, 1.8590255

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7809868, upper bound: 0.7775830
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7809868, upper bound: 0.7775830
time: 4.20 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.04 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7775826, upper bound: 0.7809890
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7775826, upper bound: 0.7809889
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7735373, upper bound: 0.7845694
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7735373, upper bound: 0.7845694
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7785245, upper bound: 0.7789644
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7785245, upper bound: 0.7789644
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7756731, upper bound: 0.7836047
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7756731, upper bound: 0.7836047
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7836048, upper bound: 0.7756754
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7836048, upper bound: 0.7756754
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7789646, upper bound: 0.7785266
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7789646, upper bound: 0.7785266
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7845701, upper bound: 0.7735373
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7845701, upper bound: 0.7735373
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7809868, upper bound: 0.7775830
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.04
Output dim: 6, lower bound: -0.7809868, upper bound: 0.7775830

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5078750, 1.5042882
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4349499, 1.4452014
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3724723, 1.3643074
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2504201, 1.2407384
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7847776, 1.7813973
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0370300, 1.0374663
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3237596, 1.3290653
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9936433, 0.9829143
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8710303, 0.8888204
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8532023, 1.8409185

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7749254, upper bound: 0.7754971
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7694409, upper bound: 0.7782868
time: 4.48 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5081587, 1.5046511
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4333735, 1.4510131
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3771539, 1.3643436
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2545733, 1.2395940
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7881260, 1.7801294
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0325506, 1.0464962
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3270283, 1.3434258
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0007372, 0.9855371
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8865743, 0.8758533
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8454757, 1.8574409

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7749254, upper bound: 0.7756520
time: 4.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7692914, upper bound: 0.7782867
time: 5.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5073147, 1.5048137
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4470520, 1.4324098
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3666682, 1.3667936
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2596946, 1.2321663
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7826633, 1.7856884
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0348406, 1.0367916
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3182683, 1.3345265
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9878201, 0.9887316
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8706467, 0.8884192
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8529944, 1.8408895

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7709626, upper bound: 0.7767639
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7675375, upper bound: 0.7818541
time: 5.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5075989, 1.5051765
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4454746, 1.4382212
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3713498, 1.3668289
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2638478, 1.2310224
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7860126, 1.7844200
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0303602, 1.0458214
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3215361, 1.3488865
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9949141, 0.9913539
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8861904, 0.8754520
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8452659, 1.8574119

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7709626, upper bound: 0.7770550
time: 7.36 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7674193, upper bound: 0.7818540
time: 4.61 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5053082, 1.5056639
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4352160, 1.4443009
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3738694, 1.3583860
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2511411, 1.2405431
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7861567, 1.7817435
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0400498, 1.0350055
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3241506, 1.3294568
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9915714, 0.9854307
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8753583, 0.8847134
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8531852, 1.8410811

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7758022, upper bound: 0.7729170
time: 4.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7709339, upper bound: 0.7764277
time: 6.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5055914, 1.5060267
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4336405, 1.4501123
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3785510, 1.3583345
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2552881, 1.2393990
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7895050, 1.7804756
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0355701, 1.0440353
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3274193, 1.3438172
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9986653, 0.9880536
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8910515, 0.8714085
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8454566, 1.8576179

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7758022, upper bound: 0.7730456
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7706857, upper bound: 0.7764278
time: 7.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5047822, 1.5061893
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4473190, 1.4332271
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3680654, 1.3666053
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2604156, 1.2325253
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7841549, 1.7860346
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0378604, 1.0377846
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3186588, 1.3349495
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9858131, 0.9912479
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8749747, 0.8850973
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8532147, 1.8410521

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7728983, upper bound: 0.7754194
time: 7.56 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7698997, upper bound: 0.7809585
time: 6.25 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5050659, 1.5065522
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4457412, 1.4390385
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3727479, 1.3665533
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2645626, 1.2313809
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7875032, 1.7847662
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0333800, 1.0468144
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3219271, 1.3493094
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9929070, 0.9938703
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8906679, 0.8717922
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8454871, 1.8575888

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7728983, upper bound: 0.7755843
time: 6.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7697605, upper bound: 0.7809578
time: 6.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5077438, 1.5053387
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4475098, 1.4463749
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3665538, 1.3861685
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2313809, 1.2838230
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8008718, 1.7854228
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0437403, 1.0328209
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3340535, 1.3211451
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9934258, 0.9832053
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8721569, 0.8880585
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8486342, 1.8423548

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7809582, upper bound: 0.7697601
time: 6.13 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7755847, upper bound: 0.7729000
time: 4.47 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5080276, 1.5057020
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4459338, 1.4521863
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3712354, 1.3862047
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2355347, 1.2826786
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8042212, 1.7841544
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0392606, 1.0418508
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3373222, 1.3355055
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0005198, 0.9858282
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8877010, 0.8750914
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8409076, 1.8588777

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7809582, upper bound: 0.7698997
time: 6.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7754197, upper bound: 0.7728985
time: 6.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5071836, 1.5058646
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4596119, 1.4325562
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3583350, 1.3886547
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2393990, 1.2752509
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7987576, 1.7874250
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0415509, 1.0315573
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3285623, 1.3266053
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9875441, 0.9890225
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8717732, 0.8876574
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8484254, 1.8423257

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7764280, upper bound: 0.7706860
time: 6.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7730457, upper bound: 0.7758043
time: 4.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5074673, 1.5062275
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4580345, 1.4383676
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3630166, 1.3886900
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2435527, 1.2741065
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8021069, 1.7861567
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0370705, 1.0405872
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3318305, 1.3409657
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9946384, 0.9916449
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8873174, 0.8746903
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8406978, 1.8588486

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7764280, upper bound: 0.7709339
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7729172, upper bound: 0.7758043
time: 4.70 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5051765, 1.5067139
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4477768, 1.4454744
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3679509, 1.3802471
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2321024, 1.2836275
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8022509, 1.7857690
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0467603, 1.0303601
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3344440, 1.3215365
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9913540, 0.9857221
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8764849, 0.8839517
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8486171, 1.8425174

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7818544, upper bound: 0.7674195
time: 6.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7770556, upper bound: 0.7709626
time: 4.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5054603, 1.5070767
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4462004, 1.4512858
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3726335, 1.3801956
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2362490, 1.2824831
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8056002, 1.7845006
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0422809, 1.0393900
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3377128, 1.3358970
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9984479, 0.9883449
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8921778, 0.8706467
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8408895, 1.8590546

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7818544, upper bound: 0.7675381
time: 6.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7767638, upper bound: 0.7709628
time: 4.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5046511, 1.5072393
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4598789, 1.4333737
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3597322, 1.3884664
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2401204, 1.2756100
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8002491, 1.7877712
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0445712, 1.0325503
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3289528, 1.3270283
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9855371, 0.9915392
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8761010, 0.8843354
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8486457, 1.8424883

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7782846, upper bound: 0.7692918
time: 8.87 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7756498, upper bound: 0.7749256
time: 4.93 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5049348, 1.5076022
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4583015, 1.4391851
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3644147, 1.3884144
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2442670, 1.2744656
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.8035975, 1.7865028
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0400908, 1.0415802
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3322210, 1.3413887
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9926310, 0.9941616
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8917940, 0.8710304
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8409190, 1.8590255

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7782846, upper bound: 0.7694409
time: 6.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7754949, upper bound: 0.7749255
time: 5.49 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 17.96 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7749254, upper bound: 0.7754971
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7694409, upper bound: 0.7782868
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7749254, upper bound: 0.7756520
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7692914, upper bound: 0.7782867
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7709626, upper bound: 0.7767639
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7675375, upper bound: 0.7818541
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7709626, upper bound: 0.7770550
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7674193, upper bound: 0.7818540
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7758022, upper bound: 0.7729170
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7709339, upper bound: 0.7764277
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7758022, upper bound: 0.7730456
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7706857, upper bound: 0.7764278
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7728983, upper bound: 0.7754194
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7698997, upper bound: 0.7809585
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7728983, upper bound: 0.7755843
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7697605, upper bound: 0.7809578
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7809582, upper bound: 0.7697601
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7755847, upper bound: 0.7729000
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7809582, upper bound: 0.7698997
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7754197, upper bound: 0.7728985
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7764280, upper bound: 0.7706860
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7730457, upper bound: 0.7758043
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7764280, upper bound: 0.7709339
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7729172, upper bound: 0.7758043
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7818544, upper bound: 0.7674195
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7770556, upper bound: 0.7709626
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7818544, upper bound: 0.7675381
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7767638, upper bound: 0.7709628
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7782846, upper bound: 0.7692918
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7756498, upper bound: 0.7749256
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7782846, upper bound: 0.7694409
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 17.96
Output dim: 6, lower bound: -0.7754949, upper bound: 0.7749255

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5058322, 1.5026689
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4409304, 1.4547312
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3677673, 1.3511057
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2418156, 1.2351990
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7541723, 1.7592888
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0267184, 1.0450628
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3430219, 1.3368154
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9932420, 0.9840000
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8645920, 0.8777816
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8493977, 1.8385015

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7643040, upper bound: 0.7645605
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7646120, upper bound: 0.7644807
time: 4.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5060563, 1.5022454
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4409180, 1.4511824
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3592691, 1.3546238
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2418089, 1.2321339
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7626829, 1.7507915
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0397079, 1.0271550
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3315096, 1.3478584
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9934905, 0.9825132
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8599917, 0.8806958
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8507862, 1.8354840

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7588174, upper bound: 0.7673113
time: 5.42 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7591360, upper bound: 0.7672450
time: 6.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5061159, 1.5030317
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4393549, 1.4605441
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3724566, 1.3511410
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2459803, 1.2340536
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7575197, 1.7581611
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0222390, 1.0526586
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3458219, 1.3509569
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0003836, 0.9870222
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8804578, 0.8648145
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8416691, 1.8543406

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7643040, upper bound: 0.7647132
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7646120, upper bound: 0.7646356
time: 4.29 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5063400, 1.5026083
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4396863, 1.4569955
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3639593, 1.3546143
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2459736, 1.2309897
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7660313, 1.7495232
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0352275, 1.0347508
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3347783, 1.3620000
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0006320, 0.9851360
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8758575, 0.8674605
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8430576, 1.8513231

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7586620, upper bound: 0.7673111
time: 7.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7590011, upper bound: 0.7672429
time: 4.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5052724, 1.5030828
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4530330, 1.4393833
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3595076, 1.3535919
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2510900, 1.2241108
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7520571, 1.7635808
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0245292, 1.0442456
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3375311, 1.3422761
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9874189, 0.9880054
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8642091, 0.8773805
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8474121, 1.8384724

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7602549, upper bound: 0.7659365
time: 5.02 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7605837, upper bound: 0.7658395
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5054970, 1.5027714
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4562678, 1.4383907
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3534660, 1.3595152
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2541537, 1.2235620
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7605677, 1.7550826
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0408483, 1.0264803
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3260179, 1.3533206
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9876673, 0.9883305
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8596081, 0.8801966
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8505764, 1.8370171

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7568266, upper bound: 0.7710218
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7571553, upper bound: 0.7709251
time: 4.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5055561, 1.5034456
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4514551, 1.4451962
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3641968, 1.3536272
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2552552, 1.2229652
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7554064, 1.7624531
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0200489, 1.0518414
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3403306, 1.3564177
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9945605, 0.9910270
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8800749, 0.8644133
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8396835, 1.8543115

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7602549, upper bound: 0.7662201
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7605837, upper bound: 0.7661258
time: 4.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5057807, 1.5031343
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4550309, 1.4442036
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3581553, 1.3595057
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2583184, 1.2224178
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7639170, 1.7538142
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0363674, 1.0340761
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3292861, 1.3674622
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9948089, 0.9909527
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8754739, 0.8669612
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8428488, 1.8528562

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7567085, upper bound: 0.7710241
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7570373, upper bound: 0.7709249
time: 5.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5032654, 1.5045881
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4412241, 1.4538577
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3704691, 1.3451843
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2423539, 1.2350035
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7555513, 1.7596350
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0287573, 1.0410128
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3434129, 1.3372068
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9911702, 0.9872262
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8694056, 0.8736748
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8491650, 1.8386626

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7651769, upper bound: 0.7619461
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7654854, upper bound: 0.7618780
time: 4.84 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5035768, 1.5041647
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4412117, 1.4502819
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3619719, 1.3513842
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2423472, 1.2319386
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7641897, 1.7511377
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0417469, 1.0246942
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3319006, 1.3482513
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9912446, 0.9857395
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8648051, 0.8778093
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8507681, 1.8356452

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7603112, upper bound: 0.7654537
time: 5.73 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7606025, upper bound: 0.7653854
time: 4.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5035491, 1.5049510
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4396486, 1.4596705
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3751583, 1.3451328
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2465186, 1.2338581
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7588987, 1.7585073
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0242779, 1.0486085
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3462129, 1.3513484
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9983118, 0.9902484
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8852712, 0.8603698
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8415852, 1.8545184

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7651769, upper bound: 0.7620706
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7654854, upper bound: 0.7620024
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5038605, 1.5045276
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4399800, 1.4560950
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3666601, 1.3511744
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2465119, 1.2307944
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7675371, 1.7498693
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0372665, 1.0322900
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3351693, 1.3623924
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9983861, 0.9883623
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8806709, 0.8649710
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8430405, 1.8515010

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7600692, upper bound: 0.7654531
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7603634, upper bound: 0.7653877
time: 4.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5027394, 1.5050020
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4533267, 1.4395397
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3622093, 1.3534031
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2516284, 1.2239156
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7535486, 1.7639265
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0265682, 1.0404618
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3379216, 1.3426991
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9854119, 0.9912317
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8690227, 0.8740585
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8476334, 1.8386331

Time for backsubstitution: 5.49 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7621902, upper bound: 0.7645880
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7625191, upper bound: 0.7644986
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5031629, 1.5046906
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4565620, 1.4392080
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3561678, 1.3620591
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2546921, 1.2239208
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7621861, 1.7554283
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0428872, 1.0274733
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3264089, 1.3537431
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9872983, 0.9915568
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8644212, 0.8781936
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8507977, 1.8371778

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7592112, upper bound: 0.7701278
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7595522, upper bound: 0.7700306
time: 5.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5030236, 1.5053649
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4517488, 1.4453528
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3668985, 1.3533516
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2557931, 1.2227700
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7568970, 1.7627993
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0220878, 1.0480576
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3407216, 1.3568411
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9925535, 0.9942533
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8848883, 0.8607535
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8400517, 1.8544888

Time for backsubstitution: 5.48 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 555

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7621902, upper bound: 0.7647610
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7625191, upper bound: 0.7646714
time: 4.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.5034466, 1.5050535
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4553246, 1.4450212
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3608570, 1.3618488
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2588568, 1.2227767
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7655344, 1.7541604
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -1.0384064, 1.0350691
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.3296771, 1.3678846
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.9944398, 0.9941790
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8802871, 0.8653538
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.8430700, 1.8530335

Time for backsubstitution: 5.47 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.3650121688842773
rel_dist={6: [-0.846481615748317, 0.8464836721348625]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2130
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: RSZ, layer: 3, pos: 2130

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7001077, upper bound: 0.7061182
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7061180, upper bound: 0.7001102
time: 4.22 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 9.75 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 9.75
Output dim: 6, lower bound: -0.7001077, upper bound: 0.7061182
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 9.75
Output dim: 6, lower bound: -0.7061180, upper bound: 0.7001102

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4499221, 1.4498239
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4468946, 1.4467850
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3559999, 1.3497500
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2197003, 1.2054210
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7587662, 1.7574644
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9830196, 0.9869454
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2712736, 1.2772150
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0579801, 1.0577731
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8184977, 0.8190690
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7591209, 1.7556949

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6953340, upper bound: 0.7014228
time: 5.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6954357, upper bound: 0.7013236
time: 4.40 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4498239, 1.4499221
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4467850, 1.4468949
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3497496, 1.3559995
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2054210, 1.2197003
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7574644, 1.7587667
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9869454, 0.9830196
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2772145, 1.2712736
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0577731, 1.0579801
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8190691, 0.8184977
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7556944, 1.7591209

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 913
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 913

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7013235, upper bound: 0.6954382
time: 4.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7014225, upper bound: 0.6953365
time: 4.18 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.05 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.05
Output dim: 6, lower bound: -0.6953340, upper bound: 0.7014228
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.05
Output dim: 6, lower bound: -0.6954357, upper bound: 0.7013236
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.05
Output dim: 6, lower bound: -0.7013235, upper bound: 0.6954382
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.05
Output dim: 6, lower bound: -0.7014225, upper bound: 0.6953365

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4504457, 1.4484477
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4459095, 1.4451866
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3545532, 1.3528094
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2189856, 1.2044373
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7573862, 1.7571182
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9800127, 0.9831936
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2708840, 1.2765074
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0565021, 1.0547903
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8141179, 0.8174340
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7585969, 1.7553368

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6857957, upper bound: 0.6893289
time: 4.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6826949, upper bound: 0.6919987
time: 5.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4485459, 1.4498239
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4468946, 1.4457998
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3559999, 1.3483028
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2197003, 1.2047062
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7584209, 1.7574644
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9830196, 0.9839385
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2712736, 1.2768245
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0549972, 1.0577731
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8184977, 0.8146890
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7587628, 1.7556949

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6865587, upper bound: 0.6873385
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6845602, upper bound: 0.6912354
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4503469, 1.4485459
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4457994, 1.4452965
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3483028, 1.3590593
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2047062, 1.2187161
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7560844, 1.7584205
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9839385, 0.9792678
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2768250, 1.2705665
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0562952, 1.0549972
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8146892, 0.8168626
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7551713, 1.7587633

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6912352, upper bound: 0.6845603
time: 4.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6873384, upper bound: 0.6865588
time: 4.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4484477, 1.4499221
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.4467850, 1.4459097
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.3497496, 1.3545532
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.2054210, 1.2189856
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7571182, 1.7587667
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9869454, 0.9800127
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2772145, 1.2708840
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -1.0547903, 1.0579801
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8190691, 0.8141177
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7553372, 1.7591209

Time for backsubstitution: 5.43 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 2480
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 2480

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6919987, upper bound: 0.6826954
time: 4.36 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6893288, upper bound: 0.6857961
time: 4.27 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 14.22 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.22
Output dim: 6, lower bound: -0.6857957, upper bound: 0.6893289
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.22
Output dim: 6, lower bound: -0.6826949, upper bound: 0.6919987
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.22
Output dim: 6, lower bound: -0.6865587, upper bound: 0.6873385
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.22
Output dim: 6, lower bound: -0.6845602, upper bound: 0.6912354
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.22
Output dim: 6, lower bound: -0.6912352, upper bound: 0.6845603
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.22
Output dim: 6, lower bound: -0.6873384, upper bound: 0.6865588
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 14.22
Output dim: 6, lower bound: -0.6919987, upper bound: 0.6826954
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 14.22
Output dim: 6, lower bound: -0.6893288, upper bound: 0.6857961

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4490561, 1.4466386
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3120360, 1.3209069
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2733130, 1.2672167
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1585083, 1.1503892
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7056675, 1.7021809
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9593933, 0.9630802
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2581697, 1.2596974
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8942475, 0.8881681
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8088703, 0.8124875
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7635417, 1.7601242

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6755423, upper bound: 0.6788551
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6755423, upper bound: 0.6788553
time: 4.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4486361, 1.4470325
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3211122, 1.3113129
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2689605, 1.2690811
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1654644, 1.1439600
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7040825, 1.7053995
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9577506, 0.9625742
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2540507, 1.2637935
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8898802, 0.8925304
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8085825, 0.8121865
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7633843, 1.7601023

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6722668, upper bound: 0.6815887
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6722668, upper bound: 0.6815887
time: 4.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4471312, 1.4480143
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3123031, 1.3202314
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2747111, 1.2627106
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1592231, 1.1502428
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7067013, 1.7025270
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9624133, 0.9612346
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2585607, 1.2599912
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8926935, 0.8906844
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8133478, 0.8091539
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7635283, 1.7603011

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6763472, upper bound: 0.6767643
time: 4.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6763472, upper bound: 0.6767643
time: 4.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4467368, 1.4484086
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3213782, 1.3119259
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2703586, 1.2688746
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1661792, 1.1442294
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7052002, 1.7057452
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9607706, 0.9633188
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2544417, 1.2641106
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8883750, 0.8950469
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8130600, 0.8094417
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7635503, 1.7602792

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6742775, upper bound: 0.6807836
time: 6.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6742775, upper bound: 0.6807836
time: 6.20 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4489579, 1.4467368
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3119264, 1.3217869
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2688746, 1.2734661
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1442294, 1.1656108
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7043648, 1.7052002
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9633191, 0.9595962
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2641106, 1.2537570
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8940845, 0.8883750
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8094416, 0.8119161
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7601151, 1.7635508

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6807838, upper bound: 0.6742797
time: 5.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6807838, upper bound: 0.6742778
time: 5.05 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4485378, 1.4471312
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3202319, 1.3114228
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2627101, 1.2735195
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1502428, 1.1582394
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7010632, 1.7067018
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9612346, 0.9586484
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2599912, 1.2578521
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8896732, 0.8926935
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8091540, 0.8116152
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7599587, 1.7635283

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6767641, upper bound: 0.6763471
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6767641, upper bound: 0.6763471
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4470325, 1.4481125
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3121929, 1.3211114
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2702727, 1.2689600
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1449442, 1.1654644
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7053995, 1.7055459
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9663389, 0.9577506
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2645016, 1.2540507
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8925304, 0.8908913
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8139191, 0.8085825
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7601027, 1.7637272

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6815883, upper bound: 0.6722691
time: 8.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6815883, upper bound: 0.6722672
time: 6.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4466381, 1.4485068
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3204985, 1.3120360
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2641082, 1.2733135
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1509571, 1.1585083
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7021809, 1.7070475
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9642544, 0.9593933
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2603822, 1.2581697
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8881681, 0.8952100
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8136313, 0.8088703
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7601247, 1.7637053

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 1502
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 1502

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6788550, upper bound: 0.6755446
time: 4.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6788550, upper bound: 0.6755446
time: 4.18 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.00 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6755423, upper bound: 0.6788551
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6755423, upper bound: 0.6788553
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6722668, upper bound: 0.6815887
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6722668, upper bound: 0.6815887
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6763472, upper bound: 0.6767643
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6763472, upper bound: 0.6767643
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6742775, upper bound: 0.6807836
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6742775, upper bound: 0.6807836
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6807838, upper bound: 0.6742797
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6807838, upper bound: 0.6742778
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6767641, upper bound: 0.6763471
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6767641, upper bound: 0.6763471
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6815883, upper bound: 0.6722691
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6815883, upper bound: 0.6722672
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6788550, upper bound: 0.6755446
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 14.00
Output dim: 6, lower bound: -0.6788550, upper bound: 0.6755446

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4483528, 1.4464765
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3164835, 1.3055015
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2642789, 1.2643728
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1613111, 1.1406646
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7007332, 1.7030020
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9520810, 0.9535444
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2372394, 1.2494330
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8827863, 0.8834698
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.7930388, 0.8063681
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7526584, 1.7435803

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6697331, upper bound: 0.6751561
time: 5.68 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6668314, upper bound: 0.6789401
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4480476, 1.4467487
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3153000, 1.3067064
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2643170, 1.2643995
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1621690, 1.1398067
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7016850, 1.7020507
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9487207, 0.9569042
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2396903, 1.2469411
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8805885, 0.8854365
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8030176, 0.7966428
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7468619, 1.7493758

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6697331, upper bound: 0.6752464
time: 5.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6668307, upper bound: 0.6789403
time: 4.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4464531, 1.4478526
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3167500, 1.3061147
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2656760, 1.2642317
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1620321, 1.1409340
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7018509, 1.7033482
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9551008, 0.9542890
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2376304, 1.2497501
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8812809, 0.8859863
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.7973666, 0.8038766
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7528243, 1.7437425

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6715022, upper bound: 0.6733278
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6695673, upper bound: 0.6781589
time: 8.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4461808, 1.4481244
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3155670, 1.3072963
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2657142, 1.2641926
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1628904, 1.1400757
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7028027, 1.7023969
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9517405, 0.9576488
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2400813, 1.2472987
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8793139, 0.8879528
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8073454, 0.7938979
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7470279, 1.7495384

Time for backsubstitution: 5.45 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6715022, upper bound: 0.6734306
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6694837, upper bound: 0.6781613
time: 5.76 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4486742, 1.4461808
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3072968, 1.3159754
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2641931, 1.2687578
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1400757, 1.1623154
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7010164, 1.7028027
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9576488, 0.9505663
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2472987, 1.2393966
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8869905, 0.8793139
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.7938981, 0.8060977
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7493892, 1.7470284

Time for backsubstitution: 5.44 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6781591, upper bound: 0.6694833
time: 6.73 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6734307, upper bound: 0.6715045
time: 4.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4483695, 1.4464531
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3061142, 1.3171818
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2642322, 1.2687845
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1409340, 1.1614575
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7019672, 1.7018514
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9542892, 0.9539262
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2497501, 1.2369056
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8847928, 0.8812809
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8038766, 0.7963723
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7435927, 1.7528243

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6781591, upper bound: 0.6695671
time: 6.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6733280, upper bound: 0.6715021
time: 5.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4467487, 1.4475565
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3075628, 1.3153000
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2655902, 1.2643175
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1407971, 1.1621690
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7020512, 1.7031488
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9606686, 0.9487207
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2476897, 1.2396903
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8854365, 0.8818302
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.7982259, 0.8030175
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7493758, 1.7471905

Time for backsubstitution: 5.46 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6789401, upper bound: 0.6668310
time: 7.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6752442, upper bound: 0.6697354
time: 4.60 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -1.6561922, 0.1125274, -1.6561922, 0.1125274, -1.4464769, 1.4478288
1: -17.9640560, -15.6267805, -17.9640560, -15.6267805, -1.3063812, 1.3164833
2: -6.5349898, -4.4621940, -6.5349898, -4.4621940, -1.2656293, 1.2642784
3: -13.9784403, -12.1074400, -13.9784403, -12.1074400, -1.1416554, 1.1613107
4: -5.6369295, -3.7163720, -5.6369295, -3.7163720, -1.7030020, 1.7021976
5: -7.0538297, -5.5888457, -7.0538297, -5.5888457, -0.9573090, 0.9520810
6: 8.2564125, 10.0458202, 8.2564125, 10.0458202, -1.2501411, 1.2372394
7: -14.0097389, -12.1174345, -14.0097389, -12.1174345, -0.8834698, 0.8837974
8: -6.1156301, -4.6512346, -6.1156301, -4.6512346, -0.8082047, 0.7930387
9: -10.8449526, -8.5048056, -10.8449526, -8.5048056, -1.7435794, 1.7529869

Time for backsubstitution: 5.47 seconds

### RS candidates at layer 3
type: RSZ, layer: 3, pos: 577
type: RSZ, layer: 3, pos: 555
type: RSZ, layer: 3, pos: 332
type: RSZ, layer: 3, pos: 2818
type: RSZ, layer: 3, pos: 660
type: RSZ, layer: 3, pos: 613
type: RSZ, layer: 3, pos: 1705
type: RSZ, layer: 3, pos: 2901
type: RSZ, layer: 3, pos: 1479
type: RSZ, layer: 3, pos: 718
type: RSZ, layer: 3, pos: 1997
type: RSZ, layer: 3, pos: 1506
type: RSZ, layer: 3, pos: 1732
type: RSZ, layer: 3, pos: 414
type: RSZ, layer: 3, pos: 2349
type: RSZ, layer: 3, pos: 2563
type: RSZ, layer: 3, pos: 1695
type: RSZ, layer: 3, pos: 2148
type: RSZ, layer: 3, pos: 75
type: RSZ, layer: 3, pos: 192
type: RSZ, layer: 3, pos: 1515
type: RSZ, layer: 3, pos: 2461
type: RSZ, layer: 3, pos: 3102
type: RSZ, layer: 3, pos: 2234
type: RSZ, layer: 3, pos: 2858
type: RSZ, layer: 3, pos: 38
type: RSZ, layer: 3, pos: 890
type: RSZ, layer: 3, pos: 1775
type: RSZ, layer: 3, pos: 886
type: RSZ, layer: 3, pos: 2396
type: RSZ, layer: 3, pos: 780
type: RSZ, layer: 3, pos: 1406
type: RSZ, layer: 3, pos: 330
type: RSZ, layer: 3, pos: 76
type: RSZ, layer: 3, pos: 1228
type: RSZ, layer: 3, pos: 738
type: RSZ, layer: 3, pos: 2827
type: RSZ, layer: 3, pos: 2917
type: RSZ, layer: 3, pos: 1979
type: RSZ, layer: 3, pos: 1847
type: RSZ, layer: 3, pos: 615
type: RSZ, layer: 3, pos: 151
type: RSZ, layer: 3, pos: 2356
type: RSZ, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 3, pos: 577

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6789401, upper bound: 0.6668317
time: 6.86 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6751555, upper bound: 0.6697330
time: 5.53 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 18.03 seconds
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6697331, upper bound: 0.6751561
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6668314, upper bound: 0.6789401
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6697331, upper bound: 0.6752464
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6668307, upper bound: 0.6789403
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6715022, upper bound: 0.6733278
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6695673, upper bound: 0.6781589
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6715022, upper bound: 0.6734306
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6694837, upper bound: 0.6781613
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6781591, upper bound: 0.6694833
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6734307, upper bound: 0.6715045
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6781591, upper bound: 0.6695671
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6733280, upper bound: 0.6715021
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6789401, upper bound: 0.6668310
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6752442, upper bound: 0.6697354
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6789401, upper bound: 0.6668317
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 18.03
Output dim: 6, lower bound: -0.6751555, upper bound: 0.6697330
Binary search (step 2): status=Status.VERIFIED, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.2817935943603516
rel_dist={6: [-0.7326686905779951, 0.7326686039477615]}

## Binary Search with RS_dual_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.01171875
execution time: 2027.98 seconds
