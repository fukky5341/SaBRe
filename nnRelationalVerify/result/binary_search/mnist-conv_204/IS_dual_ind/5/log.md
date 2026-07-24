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
execution time: IAR + LP analysis = 15.16 + 32.17 = 47.33 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.67 seconds, max iter: 100)

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
Binary search time: 209.52 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual_ind) starts
Time budget: 3343.15 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2130
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2130

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0987745, upper bound: 1.0936778
time: 3.60 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0936773, upper bound: 1.0936777
time: 3.39 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.30 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 7.30
Output dim: 6, lower bound: -1.0987745, upper bound: 1.0936778
IS_A2, status: Status.UNKNOWN, split count: 1, time: 7.30
Output dim: 6, lower bound: -1.0936773, upper bound: 1.0936777

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -1.6506550, 0.1124235, -1.6561922, 0.1125274, -1.6750298, 1.6829333
1: -17.9605484, -15.6270342, -17.9640560, -15.6267805, -1.9554849, 1.9597316
2: -6.5349741, -4.4834495, -6.5349898, -4.4621940, -1.7719660, 1.7481227
3: -13.9778080, -12.1164856, -13.9784403, -12.1074400, -1.6009030, 1.5706391
4: -5.6252713, -3.7163892, -5.6369295, -3.7163720, -1.9088993, 1.9205403
5: -7.0521522, -5.5889454, -7.0538297, -5.5888457, -1.3117192, 1.3086953
6: 8.2744751, 10.0458050, 8.2564125, 10.0458202, -1.6015840, 1.6146502
7: -14.0097361, -12.1294889, -14.0097389, -12.1174345, -1.4759142, 1.4694927
8: -6.1093316, -4.6512423, -6.1156301, -4.6512346, -1.1277351, 1.1304324
9: -10.8447342, -8.5088797, -10.8449526, -8.5048056, -2.1462402, 2.1446924

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0884659
time: 3.49 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0936776
time: 3.48 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.6475005, 0.1262470, -1.6545560, 0.1124785, -1.6745200, 1.7041864
1: -17.9519997, -15.6137323, -17.9626312, -15.6268787, -1.9853816, 1.9757223
2: -6.5853863, -4.5002308, -6.5349855, -4.4676447, -1.8386312, 1.7628379
3: -14.0151386, -12.1148701, -13.9781818, -12.1083164, -1.6408954, 1.6312375
4: -5.6131539, -3.6883640, -5.6334400, -3.7163763, -1.8967776, 1.9450760
5: -7.0425291, -5.5800605, -7.0524797, -5.5888901, -1.3461244, 1.3117352
6: 8.3123417, 10.1139135, 8.2663641, 10.0458136, -1.5765390, 1.6856112
7: -14.0357504, -12.1219559, -14.0097370, -12.1193743, -1.4979753, 1.4616508
8: -6.1018839, -4.6322112, -6.1131978, -4.6512384, -1.1163242, 1.1296991
9: -10.8561497, -8.5267410, -10.8448553, -8.5076971, -2.1452465, 2.1463761

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0936775, upper bound: 1.0884660
time: 3.42 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0936775, upper bound: 1.0936777
time: 3.62 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 12.79 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 12.79
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0884659
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 12.79
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0936776
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 12.79
Output dim: 6, lower bound: -1.0936775, upper bound: 1.0884660
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 12.79
Output dim: 6, lower bound: -1.0936775, upper bound: 1.0936777

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -1.6506550, 0.1124235, -1.6506550, 0.1124235, -1.6738825, 1.6738825
1: -17.9605484, -15.6270342, -17.9605484, -15.6270342, -1.9474187, 1.9474187
2: -6.5349741, -4.4834495, -6.5349741, -4.4834495, -1.7481084, 1.7481084
3: -13.9778080, -12.1164856, -13.9778080, -12.1164856, -1.5580115, 1.5580115
4: -5.6252713, -3.7163892, -5.6252713, -3.7163892, -1.9088821, 1.9088821
5: -7.0521522, -5.5889454, -7.0521522, -5.5889454, -1.3021502, 1.3021500
6: 8.2744751, 10.0458050, 8.2744751, 10.0458050, -1.6015654, 1.6015654
7: -14.0097361, -12.1294889, -14.0097361, -12.1294889, -1.4694903, 1.4694898
8: -6.1093316, -4.6512423, -6.1093316, -4.6512423, -1.1277282, 1.1277282
9: -10.8447342, -8.5088797, -10.8447342, -8.5088797, -2.1445794, 2.1445789

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0854552, upper bound: 1.0687506
time: 3.80 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0855019, upper bound: 1.0746142
time: 3.67 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -1.6506550, 0.1124235, -1.6475005, 0.1262470, -1.6910725, 1.6740608
1: -17.9605484, -15.6270342, -17.9519997, -15.6137323, -1.9646845, 1.9439485
2: -6.5349741, -4.4834495, -6.5853863, -4.5002308, -1.7637200, 1.8184600
3: -13.9778080, -12.1164856, -14.0151386, -12.1148701, -1.5647225, 1.5989804
4: -5.6252713, -3.7163892, -5.6131539, -3.6883640, -1.9369073, 1.8967648
5: -7.0521522, -5.5889454, -7.0425291, -5.5800605, -1.3087604, 1.2944772
6: 8.2744751, 10.0458050, 8.3123417, 10.1139135, -1.6838722, 1.5602207
7: -14.0097361, -12.1294889, -14.0357504, -12.1219559, -1.4610868, 1.4946022
8: -6.1093316, -4.6512423, -6.1018839, -4.6322112, -1.1303763, 1.1098878
9: -10.8447342, -8.5088797, -10.8561497, -8.5267410, -2.1178026, 2.1471572

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0854552, upper bound: 1.0747445
time: 3.77 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0855019, upper bound: 1.0792398
time: 3.68 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.6475005, 0.1262470, -1.6506550, 0.1124235, -1.6740608, 1.6910725
1: -17.9519997, -15.6137323, -17.9605484, -15.6270342, -1.9439483, 1.9646845
2: -6.5853863, -4.5002308, -6.5349741, -4.4834495, -1.8184600, 1.7637205
3: -14.0151386, -12.1148701, -13.9778080, -12.1164856, -1.5989804, 1.5647225
4: -5.6131539, -3.6883640, -5.6252713, -3.7163892, -1.8967648, 1.9369073
5: -7.0425291, -5.5800605, -7.0521522, -5.5889454, -1.2944775, 1.3087604
6: 8.3123417, 10.1139135, 8.2744751, 10.0458050, -1.5602212, 1.6838722
7: -14.0357504, -12.1219559, -14.0097361, -12.1294889, -1.4946020, 1.4610865
8: -6.1018839, -4.6322112, -6.1093316, -4.6512423, -1.1098878, 1.1303761
9: -10.8561497, -8.5267410, -10.8447342, -8.5088797, -2.1471572, 2.1178026

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0791859, upper bound: 1.0687509
time: 4.72 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0792398, upper bound: 1.0746143
time: 3.55 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.6475005, 0.1262470, -1.6475005, 0.1262470, -1.6968117, 1.6968117
1: -17.9519997, -15.6137323, -17.9519997, -15.6137323, -1.9858456, 1.9858458
2: -6.5853863, -4.5002308, -6.5853863, -4.5002308, -1.7687087, 1.7687082
3: -14.0151386, -12.1148701, -14.0151386, -12.1148701, -1.5967493, 1.5967493
4: -5.6131539, -3.6883640, -5.6131539, -3.6883640, -1.9247899, 1.9247899
5: -7.0425291, -5.5800605, -7.0425291, -5.5800605, -1.3443236, 1.3443234
6: 8.3123417, 10.1139135, 8.3123417, 10.1139135, -1.5944128, 1.5944130
7: -14.0357504, -12.1219559, -14.0357504, -12.1219559, -1.4854105, 1.4854109
8: -6.1018839, -4.6322112, -6.1018839, -4.6322112, -1.1150724, 1.1150723
9: -10.8561497, -8.5267410, -10.8561497, -8.5267410, -2.1458259, 2.1458254

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0791861, upper bound: 1.0687504
time: 4.23 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0792400, upper bound: 1.0746142
time: 3.64 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.61 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 6, lower bound: -1.0854552, upper bound: 1.0687506
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 6, lower bound: -1.0855019, upper bound: 1.0746142
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 6, lower bound: -1.0854552, upper bound: 1.0747445
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 6, lower bound: -1.0855019, upper bound: 1.0792398
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 6, lower bound: -1.0791859, upper bound: 1.0687509
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 6, lower bound: -1.0792398, upper bound: 1.0746143
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 6, lower bound: -1.0791861, upper bound: 1.0687504
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 13.61
Output dim: 6, lower bound: -1.0792400, upper bound: 1.0746142

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6440289, 0.1170566, -1.6496325, 0.1124226, -1.6621189, 1.6726780
1: -17.9686394, -15.6648798, -17.9605503, -15.6336317, -1.9445972, 1.9024804
2: -6.5292134, -4.4837713, -6.5340767, -4.4834991, -1.7416673, 1.7467623
3: -13.9683418, -12.1419897, -13.9778013, -12.1204376, -1.5492349, 1.5393162
4: -5.5857277, -3.7300534, -5.6191444, -3.7169027, -1.8688250, 1.8890910
5: -7.0527544, -5.6105976, -7.0517511, -5.5926695, -1.2828755, 1.2689285
6: 8.2773838, 10.0093632, 8.2744789, 10.0399609, -1.5826817, 1.5621839
7: -14.0086727, -12.1354914, -14.0095730, -12.1304045, -1.4673171, 1.4633794
8: -6.0861721, -4.6511135, -6.1056709, -4.6512423, -1.1024413, 1.1215907
9: -10.8196745, -8.5050039, -10.8408556, -8.5088797, -2.1138802, 2.1414566

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0745230, upper bound: 1.0746221
time: 3.90 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0757886, upper bound: 1.0684554
time: 4.03 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6506550, 0.1124235, -1.6635766, 1.6738648
1: -17.9605503, -15.6407909, -17.9605484, -15.6270342, -1.9473863, 1.9252341
2: -6.5333314, -4.4837227, -6.5349741, -4.4834495, -1.7466044, 1.7482853
3: -13.9777908, -12.1190071, -13.9778080, -12.1164856, -1.5580082, 1.5484080
4: -5.6204309, -3.7172384, -5.6252713, -3.7163892, -1.9040418, 1.9080329
5: -7.0514894, -5.5967188, -7.0521522, -5.5889454, -1.2994809, 1.2632952
6: 8.2744904, 10.0354328, 8.2744751, 10.0458050, -1.6015601, 1.5608320
7: -14.0094528, -12.1319809, -14.0097361, -12.1294889, -1.4682279, 1.4662898
8: -6.1021070, -4.6512437, -6.1093316, -4.6512423, -1.1096728, 1.1276909
9: -10.8361607, -8.5088787, -10.8447342, -8.5088797, -2.1302662, 2.1445794

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0852959, upper bound: 1.0926080
time: 4.01 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0852959, upper bound: 1.0928394
time: 4.03 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6440289, 0.1170566, -1.6465011, 0.1262465, -1.6793084, 1.6728468
1: -17.9686394, -15.6648798, -17.9519997, -15.6203899, -1.9618492, 1.8990099
2: -6.5292134, -4.4837713, -6.5844870, -4.5002818, -1.7572794, 1.8171206
3: -13.9683418, -12.1419897, -14.0151348, -12.1188221, -1.5559459, 1.5802836
4: -5.5857277, -3.7300534, -5.6068268, -3.6888719, -1.8968558, 1.8767734
5: -7.0527544, -5.6105976, -7.0421295, -5.5837846, -1.2896500, 1.2612641
6: 8.2773838, 10.0093632, 8.3123484, 10.1082706, -1.6651931, 1.5208395
7: -14.0086727, -12.1354914, -14.0355844, -12.1228695, -1.4588246, 1.4884992
8: -6.0861721, -4.6511135, -6.0982509, -4.6322117, -1.1050891, 1.1037436
9: -10.8196745, -8.5050039, -10.8521786, -8.5267410, -2.0871029, 2.1438704

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0673746, upper bound: 1.0633892
time: 3.90 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0689512, upper bound: 1.0587198
time: 4.01 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6475005, 0.1262470, -1.6807637, 1.6740432
1: -17.9605503, -15.6407909, -17.9519997, -15.6137323, -1.9646525, 1.9231644
2: -6.5333314, -4.4837227, -6.5853863, -4.5002308, -1.7622161, 1.8186026
3: -13.9777908, -12.1190071, -14.0151386, -12.1148701, -1.5647192, 1.5889854
4: -5.6204309, -3.7172384, -5.6131539, -3.6883640, -1.9320669, 1.8959155
5: -7.0514894, -5.5967188, -7.0425291, -5.5800605, -1.3060915, 1.2556248
6: 8.2744904, 10.0354328, 8.3123417, 10.1139135, -1.6838665, 1.5211830
7: -14.0094528, -12.1319809, -14.0357504, -12.1219559, -1.4599843, 1.4914019
8: -6.1021070, -4.6512437, -6.1018839, -4.6322112, -1.1122303, 1.1098504
9: -10.8361607, -8.5088787, -10.8561497, -8.5267410, -2.1034899, 2.1471577

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0786384, upper bound: 1.0791859
time: 3.91 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0786384, upper bound: 1.0792395
time: 4.16 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6496325, 0.1124226, -1.6622019, 1.6898656
1: -17.9600925, -15.6518440, -17.9605503, -15.6336317, -1.9425263, 1.9206121
2: -6.5795999, -4.5005484, -6.5340767, -4.4834991, -1.8120170, 1.7623763
3: -14.0056801, -12.1403723, -13.9778013, -12.1204376, -1.5897942, 1.5460262
4: -5.5723267, -3.7033584, -5.6191444, -3.7169027, -1.8554239, 1.9157860
5: -7.0431242, -5.6016593, -7.0517511, -5.5926695, -1.2752557, 1.2764525
6: 8.3133726, 10.0787506, 8.2744789, 10.0399609, -1.5430412, 1.6458015
7: -14.0346870, -12.1278963, -14.0095730, -12.1304045, -1.4924717, 1.4549847
8: -6.0788412, -4.6320820, -6.1056709, -4.6512423, -1.0846281, 1.1241479
9: -10.8304892, -8.5228643, -10.8408556, -8.5088797, -2.1153860, 2.1146798

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0610025, upper bound: 1.0675184
time: 4.96 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0632252, upper bound: 1.0621304
time: 4.92 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6506550, 0.1124235, -1.6637230, 1.6910543
1: -17.9519997, -15.6275787, -17.9605484, -15.6270342, -1.9439178, 1.9434302
2: -6.5837455, -4.5005035, -6.5349741, -4.4834495, -1.8168688, 1.7638974
3: -14.0151215, -12.1173935, -13.9778080, -12.1164856, -1.5989690, 1.5551186
4: -5.6089945, -3.6892047, -5.6252713, -3.7163892, -1.8926053, 1.9360666
5: -7.0418663, -5.5878100, -7.0521522, -5.5889454, -1.2918191, 1.2708201
6: 8.3123579, 10.1027765, 8.2744751, 10.0458050, -1.5602155, 1.6443772
7: -14.0354652, -12.1244431, -14.0097361, -12.1294889, -1.4933791, 1.4580619
8: -6.0946827, -4.6322107, -6.1093316, -4.6512423, -1.0918119, 1.1303376
9: -10.8474274, -8.5267429, -10.8447342, -8.5088797, -2.1325731, 2.1178026

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0745353, upper bound: 1.0854550
time: 5.84 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0745353, upper bound: 1.0855014
time: 5.95 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6465011, 0.1262465, -1.6849527, 1.6955948
1: -17.9600925, -15.6518440, -17.9519997, -15.6203899, -1.9822621, 1.9411125
2: -6.5795999, -4.5005484, -6.5844870, -4.5002818, -1.7622757, 1.7673731
3: -14.0056801, -12.1403723, -14.0151348, -12.1188221, -1.5889487, 1.5788660
4: -5.5723267, -3.7033584, -5.6068268, -3.6888719, -1.8834548, 1.9034684
5: -7.0431242, -5.6016593, -7.0421295, -5.5837846, -1.3235693, 1.3109925
6: 8.3133726, 10.0787506, 8.3123484, 10.1082706, -1.5751910, 1.5550473
7: -14.0346870, -12.1278963, -14.0355844, -12.1228695, -1.4831905, 1.4793155
8: -6.0788412, -4.6320820, -6.0982509, -4.6322117, -1.0897464, 1.1089158
9: -10.8304892, -8.5228643, -10.8521786, -8.5267410, -2.1151314, 2.1427059

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0610025, upper bound: 1.0575220
time: 6.53 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0632252, upper bound: 1.0526794
time: 4.20 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6475005, 0.1262470, -1.6864705, 1.6967940
1: -17.9519997, -15.6275787, -17.9519997, -15.6137323, -1.9858127, 1.9629819
2: -6.5837455, -4.5005035, -6.5853863, -4.5002308, -1.7671199, 1.7688518
3: -14.0151215, -12.1173935, -14.0151386, -12.1148701, -1.5967445, 1.5887923
4: -5.6089945, -3.6892047, -5.6131539, -3.6883640, -1.9206305, 1.9239492
5: -7.0418663, -5.5878100, -7.0425291, -5.5800605, -1.3416095, 1.3036489
6: 8.3123579, 10.1027765, 8.3123417, 10.1139135, -1.5944057, 1.5534151
7: -14.0354652, -12.1244431, -14.0357504, -12.1219559, -1.4843464, 1.4823856
8: -6.0946827, -4.6322107, -6.1018839, -4.6322112, -1.0969758, 1.1150347
9: -10.8474274, -8.5267429, -10.8561497, -8.5267410, -2.1315169, 2.1458249

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0745353, upper bound: 1.0745365
time: 5.01 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0745353, upper bound: 1.0746140
time: 6.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.80 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0745230, upper bound: 1.0746221
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0757886, upper bound: 1.0684554
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0852959, upper bound: 1.0926080
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0852959, upper bound: 1.0928394
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0673746, upper bound: 1.0633892
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0689512, upper bound: 1.0587198
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0786384, upper bound: 1.0791859
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0786384, upper bound: 1.0792395
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0610025, upper bound: 1.0675184
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0632252, upper bound: 1.0621304
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0745353, upper bound: 1.0854550
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0745353, upper bound: 1.0855014
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0610025, upper bound: 1.0575220
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0632252, upper bound: 1.0526794
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0745353, upper bound: 1.0745365
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.80
Output dim: 6, lower bound: -1.0745353, upper bound: 1.0746140

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6438589, 0.1161822, -1.6520731, 0.1031385, -1.6481247, 1.6725726
1: -17.9686317, -15.6681881, -17.9641266, -15.6691675, -1.9080648, 1.8990042
2: -6.5288963, -4.4839773, -6.5306892, -4.4790859, -1.7429481, 1.7392979
3: -13.9682255, -12.1428385, -13.9849882, -12.1295509, -1.5374732, 1.5434065
4: -5.5802670, -3.7302136, -5.5624719, -3.7323999, -1.8478670, 1.8322582
5: -7.0525498, -5.6114335, -7.0623093, -5.6016283, -1.2701585, 1.2764406
6: 8.2791271, 10.0093517, 8.2927876, 10.0546913, -1.5808229, 1.5331478
7: -14.0085640, -12.1355600, -14.0084648, -12.1311369, -1.4659238, 1.4625211
8: -6.0857420, -4.6530190, -6.1001935, -4.6709433, -1.0796385, 1.1122478
9: -10.8178778, -8.5051088, -10.8218365, -8.5148382, -2.1060071, 2.1254492

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0542196, upper bound: 1.0401142
time: 3.79 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0615940, upper bound: 1.0605353
time: 3.93 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.6440289, 0.1170566, -1.6485811, 0.1040424, -1.6583076, 1.6676893
1: -17.9686394, -15.6648798, -17.9602833, -15.6537380, -1.9380822, 1.8994119
2: -6.5292134, -4.4837713, -6.5240536, -4.4849262, -1.7409458, 1.7396655
3: -13.9683418, -12.1419897, -13.9770184, -12.1323032, -1.5439029, 1.5387254
4: -5.5857277, -3.7300534, -5.6117249, -3.7177689, -1.8679588, 1.8816714
5: -7.0527544, -5.6105976, -7.0505390, -5.6101704, -1.2774496, 1.2680416
6: 8.2773838, 10.0093632, 8.2976522, 10.0398808, -1.5826087, 1.5337162
7: -14.0086727, -12.1354914, -14.0088320, -12.1307831, -1.4664323, 1.4633973
8: -6.0861721, -4.6511135, -6.1035018, -4.6585503, -1.0950403, 1.1165395
9: -10.8196745, -8.5050039, -10.8378792, -8.5093870, -2.1058455, 2.1549296

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0753420, upper bound: 1.0672401
time: 4.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0753420, upper bound: 1.0684565
time: 3.84 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6440289, 0.1170566, -1.6716490, 1.6621037
1: -17.9605503, -15.6407909, -17.9686394, -15.6648798, -1.9024520, 1.9416087
2: -6.5333314, -4.4837227, -6.5292134, -4.4837713, -1.7463398, 1.7413907
3: -13.9777908, -12.1190071, -13.9683418, -12.1419897, -1.5393143, 1.5497470
4: -5.6204309, -3.7172384, -5.5857277, -3.7300534, -1.8903775, 1.8684893
5: -7.0514894, -5.5967188, -7.0527544, -5.6105976, -1.2668731, 1.2867591
6: 8.2744904, 10.0354328, 8.2773838, 10.0093632, -1.5621805, 1.5870953
7: -14.0094528, -12.1319809, -14.0086727, -12.1354914, -1.4634690, 1.4649630
8: -6.1021070, -4.6512437, -6.0861721, -4.6511135, -1.1194965, 1.1024054
9: -10.8361607, -8.5088787, -10.8196745, -8.5050039, -2.1400642, 2.1138802

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0746223, upper bound: 1.0745227
time: 4.10 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0684552, upper bound: 1.0757903
time: 4.02 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6450552, 0.1124195, -1.6635604, 1.6635604
1: -17.9605503, -15.6407909, -17.9605503, -15.6407909, -1.9252148, 1.9252148
2: -6.5333314, -4.4837227, -6.5333314, -4.4837227, -1.7467809, 1.7467809
3: -13.9777908, -12.1190071, -13.9777908, -12.1190071, -1.5484066, 1.5484066
4: -5.6204309, -3.7172384, -5.6204309, -3.7172384, -1.9031925, 1.9031925
5: -7.0514894, -5.5967188, -7.0514894, -5.5967188, -1.2613907, 1.2613907
6: 8.2744904, 10.0354328, 8.2744904, 10.0354328, -1.5608273, 1.5608275
7: -14.0094528, -12.1319809, -14.0094528, -12.1319809, -1.4651265, 1.4651265
8: -6.1021070, -4.6512437, -6.1021070, -4.6512437, -1.1096561, 1.1096561
9: -10.8361607, -8.5088787, -10.8361607, -8.5088787, -2.1302662, 2.1302662

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0746223, upper bound: 1.0747565
time: 4.09 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0684552, upper bound: 1.0761113
time: 4.55 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6438589, 0.1161822, -1.6486853, 0.1169801, -1.6652975, 1.6725307
1: -17.9686317, -15.6681881, -17.9555759, -15.6562576, -1.9248915, 1.8954971
2: -6.5288963, -4.4839773, -6.5811000, -4.4958296, -1.7585349, 1.8096547
3: -13.9682255, -12.1428385, -14.0223885, -12.1279354, -1.5441666, 1.5842056
4: -5.5802670, -3.7302136, -5.5502539, -3.7043874, -1.8758795, 1.8200402
5: -7.0525498, -5.6114335, -7.0527010, -5.5926938, -1.2771807, 1.2688179
6: 8.2791271, 10.0093517, 8.3304272, 10.1230049, -1.6633291, 1.4920657
7: -14.0085640, -12.1355600, -14.0344858, -12.1235609, -1.4575031, 1.4876552
8: -6.0857420, -4.6530190, -6.0927286, -4.6519084, -1.0822983, 1.0942273
9: -10.8178778, -8.5051088, -10.8332672, -8.5326490, -2.0792360, 2.1279225

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0460772, upper bound: 1.0268153
time: 3.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0544628, upper bound: 1.0489624
time: 4.11 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6440289, 0.1170566, -1.6454349, 0.1178841, -1.6754971, 1.6676574
1: -17.9686394, -15.6648798, -17.9517345, -15.6396084, -1.9549005, 1.8959069
2: -6.5292134, -4.4837713, -6.5744643, -4.5016809, -1.7565603, 1.8100228
3: -13.9683418, -12.1419897, -14.0143290, -12.1306896, -1.5506077, 1.5796480
4: -5.5857277, -3.7300534, -5.5993781, -3.6897464, -1.8959813, 1.8693247
5: -7.0527544, -5.6105976, -7.0409141, -5.6007814, -1.2844443, 1.2604198
6: 8.2773838, 10.0093632, 8.3357859, 10.1081924, -1.6651173, 1.4923668
7: -14.0086727, -12.1354914, -14.0348492, -12.1232204, -1.4579756, 1.4885280
8: -6.0861721, -4.6511135, -6.0961437, -4.6395154, -1.0976996, 1.0984313
9: -10.8196745, -8.5050039, -10.8492479, -8.5272217, -2.0790725, 2.1573997

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0683367, upper bound: 1.0565620
time: 4.14 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0683367, upper bound: 1.0587215
time: 4.44 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6410147, 0.1308787, -1.6888366, 1.6621866
1: -17.9605503, -15.6407909, -17.9600925, -15.6518440, -1.9205837, 1.9395375
2: -6.5333314, -4.4837227, -6.5795999, -4.5005484, -1.7619543, 1.8117404
3: -13.9777908, -12.1190071, -14.0056801, -12.1403723, -1.5460243, 1.5903068
4: -5.6204309, -3.7172384, -5.5723267, -3.7033584, -1.9170725, 1.8550882
5: -7.0514894, -5.5967188, -7.0431242, -5.6016593, -1.2743974, 1.2791390
6: 8.2744904, 10.0354328, 8.3133726, 10.0787506, -1.6457982, 1.5474548
7: -14.0094528, -12.1319809, -14.0346870, -12.1278963, -1.4550743, 1.4901173
8: -6.1021070, -4.6512437, -6.0788412, -4.6320820, -1.1220539, 1.0845921
9: -10.8361607, -8.5088787, -10.8304892, -8.5228643, -2.1132874, 2.1153860

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0675039, upper bound: 1.0610017
time: 3.87 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0620969, upper bound: 1.0632268
time: 4.04 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6419480, 0.1262433, -1.6807485, 1.6637063
1: -17.9605503, -15.6407909, -17.9519997, -15.6275787, -1.9434104, 1.9231453
2: -6.5333314, -4.4837227, -6.5837455, -4.5005035, -1.7623930, 1.8170118
3: -13.9777908, -12.1190071, -14.0151215, -12.1173935, -1.5551167, 1.5889750
4: -5.6204309, -3.7172384, -5.6089945, -3.6892047, -1.9312263, 1.8917561
5: -7.0514894, -5.5967188, -7.0418663, -5.5878100, -1.2689157, 1.2537315
6: 8.2744904, 10.0354328, 8.3123579, 10.1027765, -1.6443725, 1.5211780
7: -14.0094528, -12.1319809, -14.0354652, -12.1244431, -1.4569592, 1.4902773
8: -6.1021070, -4.6512437, -6.0946827, -4.6322107, -1.1122134, 1.0917950
9: -10.8361607, -8.5088787, -10.8474274, -8.5267429, -2.1034889, 2.1325727

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0675039, upper bound: 1.0610549
time: 3.85 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0620969, upper bound: 1.0633206
time: 3.71 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6408455, 0.1300051, -1.6520731, 0.1031385, -1.6482034, 1.6897573
1: -17.9600792, -15.6552572, -17.9641266, -15.6691675, -1.9059930, 1.9169676
2: -6.5792809, -4.5007534, -6.5306892, -4.4790859, -1.8132982, 1.7549157
3: -14.0055599, -12.1412230, -13.9849882, -12.1295509, -1.5780272, 1.5501161
4: -5.5668731, -3.7035234, -5.5624719, -3.7323999, -1.8344731, 1.8589485
5: -7.0429173, -5.6024947, -7.0623093, -5.6016283, -1.2625468, 1.2839668
6: 8.3151035, 10.0787392, 8.2927876, 10.0546913, -1.5412230, 1.6167650
7: -14.0345783, -12.1279573, -14.0084648, -12.1311369, -1.4910789, 1.4541314
8: -6.0784225, -4.6339841, -6.1001935, -4.6709433, -1.0618482, 1.1148056
9: -10.8286991, -8.5229626, -10.8218365, -8.5148382, -2.1075187, 2.0986729

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0409052, upper bound: 1.0326150
time: 4.14 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0472816, upper bound: 1.0540300
time: 6.26 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6485811, 0.1040424, -1.6582360, 1.6848769
1: -17.9600925, -15.6518440, -17.9602833, -15.6537380, -1.9360127, 1.9175434
2: -6.5795999, -4.5005484, -6.5240536, -4.4849262, -1.8112955, 1.7552800
3: -14.0056801, -12.1403723, -13.9770184, -12.1323032, -1.5843511, 1.5454354
4: -5.5723267, -3.7033584, -5.6117249, -3.7177689, -1.8545578, 1.9083664
5: -7.0431242, -5.6016593, -7.0505390, -5.6101704, -1.2697787, 1.2755659
6: 8.3133726, 10.0787506, 8.2976522, 10.0398808, -1.5429683, 1.6173372
7: -14.0346870, -12.1278963, -14.0088320, -12.1307831, -1.4915874, 1.4550025
8: -6.0788412, -4.6320820, -6.1035018, -4.6585503, -1.0769367, 1.1190965
9: -10.8304892, -8.5228643, -10.8378792, -8.5093870, -2.1073523, 2.1281528

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0626097, upper bound: 1.0606059
time: 5.74 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0626097, upper bound: 1.0621302
time: 7.33 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6440289, 0.1170566, -1.6718116, 1.6792932
1: -17.9519997, -15.6275787, -17.9686394, -15.6648798, -1.8989835, 1.9580626
2: -6.5837455, -4.5005035, -6.5292134, -4.4837713, -1.8166046, 1.7570028
3: -14.0151215, -12.1173935, -13.9683418, -12.1419897, -1.5802746, 1.5564594
4: -5.6089945, -3.6892047, -5.5857277, -3.7300534, -1.8789411, 1.8965230
5: -7.0418663, -5.5878100, -7.0527544, -5.6105976, -1.2592115, 1.2928598
6: 8.3123579, 10.1027765, 8.2773838, 10.0093632, -1.5208359, 1.6686482
7: -14.0354652, -12.1244431, -14.0086727, -12.1354914, -1.4885592, 1.4567351
8: -6.0946827, -4.6322107, -6.0861721, -4.6511135, -1.1016828, 1.1050520
9: -10.8474274, -8.5267429, -10.8196745, -8.5050039, -2.1423769, 2.0871029

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0631797, upper bound: 1.0673764
time: 3.87 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0689530
time: 4.61 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6450552, 0.1124195, -1.6637063, 1.6807485
1: -17.9519997, -15.6275787, -17.9605503, -15.6407909, -1.9231453, 1.9434106
2: -6.5837455, -4.5005035, -6.5333314, -4.4837227, -1.8170118, 1.7623930
3: -14.0151215, -12.1173935, -13.9777908, -12.1190071, -1.5889750, 1.5551171
4: -5.6089945, -3.6892047, -5.6204309, -3.7172384, -1.8917561, 1.9312263
5: -7.0418663, -5.5878100, -7.0514894, -5.5967188, -1.2537315, 1.2689157
6: 8.3123579, 10.1027765, 8.2744904, 10.0354328, -1.5211782, 1.6443725
7: -14.0354652, -12.1244431, -14.0094528, -12.1319809, -1.4902778, 1.4569595
8: -6.0946827, -4.6322107, -6.1021070, -4.6512437, -1.0917950, 1.1122133
9: -10.8474274, -8.5267429, -10.8361607, -8.5088787, -2.1325722, 2.1034894

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0631797, upper bound: 1.0674225
time: 7.30 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0691045
time: 3.72 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6408455, 0.1300051, -1.6486853, 0.1169801, -1.6709361, 1.6952758
1: -17.9600792, -15.6552572, -17.9555759, -15.6562576, -1.9455795, 1.9375167
2: -6.5792809, -4.5007534, -6.5811000, -4.4958296, -1.7635670, 1.7599125
3: -14.0055599, -12.1412230, -14.0223885, -12.1279354, -1.5772409, 1.5828667
4: -5.5668731, -3.7035234, -5.5502539, -3.7043874, -1.8624856, 1.8467305
5: -7.0429173, -5.6024947, -7.0527010, -5.5926938, -1.3110352, 1.3184676
6: 8.3151035, 10.0787392, 8.3304272, 10.1230049, -1.5733051, 1.5256851
7: -14.0345783, -12.1279573, -14.0344858, -12.1235609, -1.4818692, 1.4784770
8: -6.0784225, -4.6339841, -6.0927286, -4.6519084, -1.0669490, 1.0991808
9: -10.8286991, -8.5229626, -10.8332672, -8.5326490, -2.1072583, 2.1266713

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0409052, upper bound: 1.0212296
time: 3.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0472816, upper bound: 1.0435771
time: 6.00 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6454349, 0.1178841, -1.6809859, 1.6904049
1: -17.9600925, -15.6518440, -17.9517345, -15.6396084, -1.9754605, 1.9380620
2: -6.5795999, -4.5005484, -6.5744643, -4.5016809, -1.7615862, 1.7602754
3: -14.0056801, -12.1403723, -14.0143290, -12.1306896, -1.5835414, 1.5782886
4: -5.5723267, -3.7033584, -5.5993781, -3.6897464, -1.8825803, 1.8960197
5: -7.0431242, -5.6016593, -7.0409141, -5.6007814, -1.3182774, 1.3101104
6: 8.3133726, 10.0787506, 8.3357859, 10.1081924, -1.5751181, 1.5262909
7: -14.0346870, -12.1278963, -14.0348492, -12.1232204, -1.4823415, 1.4793439
8: -6.0788412, -4.6320820, -6.0961437, -4.6395154, -1.0820607, 1.1034901
9: -10.8304892, -8.5228643, -10.8492479, -8.5272217, -2.1070967, 2.1561756

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0626097, upper bound: 1.0506519
time: 4.75 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0626097, upper bound: 1.0526794
time: 4.06 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6410147, 0.1308787, -1.6945586, 1.6849380
1: -17.9519997, -15.6275787, -17.9600925, -15.6518440, -1.9410839, 1.9796486
2: -6.5837455, -4.5005035, -6.5795999, -4.5005484, -1.7668581, 1.7619991
3: -14.0151215, -12.1173935, -14.0056801, -12.1403723, -1.5788627, 1.5888963
4: -5.6089945, -3.6892047, -5.5723267, -3.7033584, -1.9056361, 1.8831220
5: -7.0418663, -5.5878100, -7.0431242, -5.6016593, -1.3088717, 1.3275008
6: 8.3123579, 10.1027765, 8.3133726, 10.0787506, -1.5550423, 1.5795908
7: -14.0354652, -12.1244431, -14.0346870, -12.1278963, -1.4793754, 1.4811008
8: -6.0946827, -4.6322107, -6.0788412, -4.6320820, -1.1068478, 1.0897108
9: -10.8474274, -8.5267429, -10.8304892, -8.5228643, -2.1413202, 2.1151314

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0631797, upper bound: 1.0563876
time: 3.55 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0583663
time: 3.82 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6419480, 0.1262433, -1.6864543, 1.6864543
1: -17.9519997, -15.6275787, -17.9519997, -15.6275787, -1.9629607, 1.9629607
2: -6.5837455, -4.5005035, -6.5837455, -4.5005035, -1.7672629, 1.7672629
3: -14.0151215, -12.1173935, -14.0151215, -12.1173935, -1.5887885, 1.5887887
4: -5.6089945, -3.6892047, -5.6089945, -3.6892047, -1.9197898, 1.9197898
5: -7.0418663, -5.5878100, -7.0418663, -5.5878100, -1.3017001, 1.3016999
6: 8.3123579, 10.1027765, 8.3123579, 10.1027765, -1.5534091, 1.5534091
7: -14.0354652, -12.1244431, -14.0354652, -12.1244431, -1.4813204, 1.4813209
8: -6.0946827, -4.6322107, -6.0946827, -4.6322107, -1.0969592, 1.0969594
9: -10.8474274, -8.5267429, -10.8474274, -8.5267429, -2.1315169, 2.1315169

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0631797, upper bound: 1.0564494
time: 4.90 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0585298
time: 4.87 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.58 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0542196, upper bound: 1.0401142
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0615940, upper bound: 1.0605353
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0753420, upper bound: 1.0672401
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0753420, upper bound: 1.0684565
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0746223, upper bound: 1.0745227
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0684552, upper bound: 1.0757903
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0746223, upper bound: 1.0747565
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0684552, upper bound: 1.0761113
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0460772, upper bound: 1.0268153
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0544628, upper bound: 1.0489624
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0683367, upper bound: 1.0565620
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0683367, upper bound: 1.0587215
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0675039, upper bound: 1.0610017
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0620969, upper bound: 1.0632268
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0675039, upper bound: 1.0610549
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0620969, upper bound: 1.0633206
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0409052, upper bound: 1.0326150
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0472816, upper bound: 1.0540300
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0626097, upper bound: 1.0606059
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0626097, upper bound: 1.0621302
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0631797, upper bound: 1.0673764
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0689530
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0631797, upper bound: 1.0674225
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0691045
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0409052, upper bound: 1.0212296
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0472816, upper bound: 1.0435771
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0626097, upper bound: 1.0506519
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0626097, upper bound: 1.0526794
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0631797, upper bound: 1.0563876
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0583663
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0631797, upper bound: 1.0564494
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.58
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0585298

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6393421, 0.0841632, -1.6509105, 0.0977132, -1.6462469, 1.6444163
1: -17.9011211, -15.6555967, -17.9474869, -15.6737080, -1.7875104, 1.7746562
2: -6.5206141, -4.5895071, -6.5286012, -4.4970407, -1.7057185, 1.6262174
3: -13.9078960, -12.1593761, -13.9741182, -12.1296263, -1.4785342, 1.5193326
4: -5.5255632, -3.7289000, -5.5522261, -3.7351825, -1.7903807, 1.8233261
5: -7.0161028, -5.5973148, -7.0548182, -5.6016545, -1.2160900, 1.2706621
6: 8.3112612, 10.0109644, 8.2983789, 10.0525513, -1.5178008, 1.5115829
7: -14.0025787, -12.2628727, -14.0084400, -12.1545124, -1.3965454, 1.3057272
8: -6.0360837, -4.6391540, -6.0911760, -4.6709685, -1.0283321, 1.1171846
9: -10.7521582, -8.5077267, -10.8082333, -8.5152674, -2.0170112, 2.0901675

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0472748, upper bound: 1.0401142
time: 4.01 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0472748, upper bound: 1.0401138
time: 5.72 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6427879, 0.1097240, -1.6520731, 0.1031385, -1.6471329, 1.6638298
1: -17.9398079, -15.6727486, -17.9641266, -15.6691675, -1.7803967, 1.8689046
2: -6.5271826, -4.5048037, -6.5306892, -4.4790859, -1.7411890, 1.6588097
3: -13.9636621, -12.1431456, -13.9849882, -12.1295509, -1.4879212, 1.5430112
4: -5.5615273, -3.7330251, -5.5624719, -3.7323999, -1.8291273, 1.8294468
5: -7.0380507, -5.6117430, -7.0623093, -5.6016283, -1.2384210, 1.2760971
6: 8.3045683, 10.0075026, 8.2927876, 10.0546913, -1.5502691, 1.5296116
7: -14.0081558, -12.1556969, -14.0084648, -12.1311369, -1.4652739, 1.2999146
8: -6.0655947, -4.6534371, -6.1001935, -4.6709433, -1.0576091, 1.1116492
9: -10.8029671, -8.5056734, -10.8218365, -8.5148382, -2.0761285, 2.1287732

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0542874, upper bound: 1.0605371
time: 3.94 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0542874, upper bound: 1.0605371
time: 3.76 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6464213, 0.1077677, -1.6485811, 0.1040424, -1.6545339, 1.6543307
1: -17.9722176, -15.7002621, -17.9602833, -15.6537380, -1.9262018, 1.8631771
2: -6.5258260, -4.4793563, -6.5240536, -4.4849262, -1.7335968, 1.7416582
3: -13.9755287, -12.1511011, -13.9770184, -12.1323032, -1.5455804, 1.5270519
4: -5.5290546, -3.7453995, -5.6117249, -3.7177689, -1.8112857, 1.8663254
5: -7.0632620, -5.6195698, -7.0505390, -5.6101704, -1.2800412, 1.2554641
6: 8.2957325, 10.0240555, 8.2976522, 10.0398808, -1.5536089, 1.5448050
7: -14.0075684, -12.1362238, -14.0088320, -12.1307831, -1.4657562, 1.4609272
8: -6.0807619, -4.6708126, -6.1035018, -4.6585503, -1.0889132, 1.0943474
9: -10.8006496, -8.5109739, -10.8378792, -8.5093870, -2.0909386, 2.1231503

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0672396
time: 5.21 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0672402
time: 5.71 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6429741, 0.1086736, -1.6485811, 0.1040424, -1.6544085, 1.6650538
1: -17.9683800, -15.6852322, -17.9602833, -15.6537380, -1.9361730, 1.8942237
2: -6.5191884, -4.4851971, -6.5240536, -4.4849262, -1.7338119, 1.7389426
3: -13.9675598, -12.1538582, -13.9770184, -12.1323032, -1.5433087, 1.5333834
4: -5.5783086, -3.7308555, -5.6117249, -3.7177689, -1.8605397, 1.8808694
5: -7.0515852, -5.6281075, -7.0505390, -5.6101704, -1.2765117, 1.2624822
6: 8.3005772, 10.0092840, 8.2976522, 10.0398808, -1.5542407, 1.5336366
7: -14.0079336, -12.1358719, -14.0088320, -12.1307831, -1.4664497, 1.4625125
8: -6.0840020, -4.6584201, -6.1035018, -4.6585503, -1.0910568, 1.1104685
9: -10.8166981, -8.5055161, -10.8378792, -8.5093870, -2.1204400, 2.1478939

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0684570
time: 4.12 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0684570
time: 4.08 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6474874, 0.1031362, -1.6438589, 0.1161822, -1.6715345, 1.6481099
1: -17.9641285, -15.6763124, -17.9686317, -15.6681881, -1.8989758, 1.9051473
2: -6.5299435, -4.4793100, -6.5288963, -4.4839773, -1.7388802, 1.7426710
3: -13.9849815, -12.1281176, -13.9682255, -12.1428385, -1.5434036, 1.5379820
4: -5.5637589, -3.7327278, -5.5802670, -3.7302136, -1.8335452, 1.8475392
5: -7.0620365, -5.6056719, -7.0525498, -5.6114335, -1.2743883, 1.2740479
6: 8.2927990, 10.0501604, 8.2791271, 10.0093517, -1.5331454, 1.5852284
7: -14.0083485, -12.1327143, -14.0085640, -12.1355600, -1.4626102, 1.4635570
8: -6.0966654, -4.6709442, -6.0857420, -4.6530190, -1.1101699, 1.0796030
9: -10.8171368, -8.5148392, -10.8178778, -8.5051088, -2.1240711, 2.1060066

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0401139, upper bound: 1.0542197
time: 3.88 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0605354, upper bound: 1.0615938
time: 3.97 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6440014, 0.1040402, -1.6440289, 0.1170566, -1.6666560, 1.6582928
1: -17.9602852, -15.6609287, -17.9686394, -15.6648798, -1.8993835, 1.9351466
2: -6.5233088, -4.4851484, -6.5292134, -4.4837713, -1.7392430, 1.7406678
3: -13.9770088, -12.1308727, -13.9683418, -12.1419897, -1.5387230, 1.5444140
4: -5.6130128, -3.7180994, -5.5857277, -3.7300534, -1.8829594, 1.8676283
5: -7.0502768, -5.6142168, -7.0527544, -5.6105976, -1.2659857, 1.2813358
6: 8.2976637, 10.0353537, 8.2773838, 10.0093632, -1.5337136, 1.5870223
7: -14.0087109, -12.1323614, -14.0086727, -12.1354914, -1.4634855, 1.4640713
8: -6.0999498, -4.6585493, -6.0861721, -4.6511135, -1.1144687, 1.0950052
9: -10.8331833, -8.5093851, -10.8196745, -8.5050039, -2.1535649, 2.1058459

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0753438
time: 4.17 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0757883
time: 4.76 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6474874, 0.1031362, -1.6448870, 0.1115442, -1.6634302, 1.6495590
1: -17.9641285, -15.6763124, -17.9605408, -15.6442041, -1.9217248, 1.8886697
2: -6.5299435, -4.4793100, -6.5330138, -4.4839301, -1.7393208, 1.7480497
3: -13.9849815, -12.1281176, -13.9776754, -12.1198549, -1.5524988, 1.5366397
4: -5.5637589, -3.7327278, -5.6149707, -3.7174134, -1.8463454, 1.8822429
5: -7.0620365, -5.6056719, -7.0512733, -5.5975542, -1.2689815, 1.2486529
6: 8.2927990, 10.0501604, 8.2762299, 10.0354204, -1.5318170, 1.5589609
7: -14.0083485, -12.1327143, -14.0093441, -12.1320486, -1.4642668, 1.4637218
8: -6.0966654, -4.6709442, -6.1016784, -4.6531477, -1.1002824, 1.0868487
9: -10.8171368, -8.5148392, -10.8343620, -8.5089817, -2.1142550, 2.1224251

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0401139, upper bound: 1.0542199
time: 3.62 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0612062, upper bound: 1.0619329
time: 4.03 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6440014, 0.1040402, -1.6450552, 0.1124195, -1.6585512, 1.6598163
1: -17.9602852, -15.6609287, -17.9605503, -15.6407909, -1.9221315, 1.9187112
2: -6.5233088, -4.4851484, -6.5333314, -4.4837227, -1.7396832, 1.7460580
3: -13.9770088, -12.1308727, -13.9777908, -12.1190071, -1.5478172, 1.5430713
4: -5.6130128, -3.7180994, -5.6204309, -3.7172384, -1.8957744, 1.9023316
5: -7.0502768, -5.6142168, -7.0514894, -5.5967188, -1.2605417, 1.2559290
6: 8.2976637, 10.0353537, 8.2744904, 10.0354328, -1.5324588, 1.5607543
7: -14.0087109, -12.1323614, -14.0094528, -12.1319809, -1.4651442, 1.4642351
8: -6.0999498, -4.6585493, -6.1021070, -4.6512437, -1.1046042, 1.1023982
9: -10.8331833, -8.5093851, -10.8361607, -8.5088787, -2.1437540, 2.1222391

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0678214, upper bound: 1.0756495
time: 3.86 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0678214, upper bound: 1.0761132
time: 3.77 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6393421, 0.0841632, -1.6476055, 0.1115509, -1.6634469, 1.6443133
1: -17.9011211, -15.6555967, -17.9389343, -15.6609306, -1.8042002, 1.7711334
2: -6.5206141, -4.5895071, -6.5789847, -4.5133257, -1.7216158, 1.6965179
3: -13.9078960, -12.1593761, -14.0115700, -12.1280107, -1.4852290, 1.5602877
4: -5.5255632, -3.7289000, -5.5404778, -3.7071626, -1.8184006, 1.8115778
5: -7.0161028, -5.5973148, -7.0452003, -5.5927320, -1.2230973, 1.2632210
6: 8.3112612, 10.0109644, 8.3354931, 10.1207848, -1.6002374, 1.4707823
7: -14.0025787, -12.2628727, -14.0344591, -12.1464720, -1.3882504, 1.3308644
8: -6.0360837, -4.6391540, -6.0836658, -4.6519337, -1.0309949, 1.0991484
9: -10.7521582, -8.5077267, -10.8192530, -8.5330505, -1.9902983, 2.0919452

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0393675, upper bound: 1.0268148
time: 3.89 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0393675, upper bound: 1.0268150
time: 5.19 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6427879, 0.1097240, -1.6486853, 0.1169801, -1.6643057, 1.6637888
1: -17.9398079, -15.6727486, -17.9555759, -15.6562576, -1.7980366, 1.8653975
2: -6.5271826, -4.5048037, -6.5811000, -4.4958296, -1.7567759, 1.7306156
3: -13.9636621, -12.1431456, -14.0223885, -12.1279354, -1.4947701, 1.5838103
4: -5.5615273, -3.7330251, -5.5502539, -3.7043874, -1.8571398, 1.8172288
5: -7.0380507, -5.6117430, -7.0527010, -5.5926938, -1.2455950, 1.2684743
6: 8.3045683, 10.0075026, 8.3304272, 10.1230049, -1.6336374, 1.4885292
7: -14.0081558, -12.1556969, -14.0344858, -12.1235609, -1.4568532, 1.3239429
8: -6.0655947, -4.6534371, -6.0927286, -4.6519084, -1.0609733, 1.0936286
9: -10.8029671, -8.5056734, -10.8332672, -8.5326490, -2.0493574, 2.1320806

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0477372, upper bound: 1.0489623
time: 4.16 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0477372, upper bound: 1.0489617
time: 4.76 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6464213, 0.1077677, -1.6454349, 0.1178841, -1.6716938, 1.6542983
1: -17.9722176, -15.7002621, -17.9517345, -15.6396084, -1.9436116, 1.8596721
2: -6.5258260, -4.4793563, -6.5744643, -4.5016809, -1.7492113, 1.8120165
3: -13.9755287, -12.1511011, -14.0143290, -12.1306896, -1.5522838, 1.5679746
4: -5.5290546, -3.7453995, -5.5993781, -3.6897464, -1.8393083, 1.8539786
5: -7.0632620, -5.6195698, -7.0409141, -5.6007814, -1.2867062, 1.2478421
6: 8.2957325, 10.0240555, 8.3357859, 10.1081924, -1.6361175, 1.5037947
7: -14.0075684, -12.1362238, -14.0348492, -12.1232204, -1.4572985, 1.4860580
8: -6.0807619, -4.6708126, -6.0961437, -4.6395154, -1.0915699, 1.0762393
9: -10.8006496, -8.5109739, -10.8492479, -8.5272217, -2.0641646, 2.1256056

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0605684, upper bound: 1.0565615
time: 4.88 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0605684, upper bound: 1.0565620
time: 5.73 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6429741, 0.1086736, -1.6454349, 0.1178841, -1.6715980, 1.6650395
1: -17.9683800, -15.6852322, -17.9517345, -15.6396084, -1.9529908, 1.8907189
2: -6.5191884, -4.4851971, -6.5744643, -4.5016809, -1.7494264, 1.8092995
3: -13.9675598, -12.1538582, -14.0143290, -12.1306896, -1.5500131, 1.5741949
4: -5.5783086, -3.7308555, -5.5993781, -3.6897464, -1.8885622, 1.8685226
5: -7.0515852, -5.6281075, -7.0409141, -5.6007814, -1.2835066, 1.2548091
6: 8.3005772, 10.0092840, 8.3357859, 10.1081924, -1.6367469, 1.4922872
7: -14.0079336, -12.1358719, -14.0348492, -12.1232204, -1.4579921, 1.4876432
8: -6.0840020, -4.6584201, -6.0961437, -4.6395154, -1.0937159, 1.0923780
9: -10.8166981, -8.5055161, -10.8492479, -8.5272217, -2.0936646, 2.1503644

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0605684, upper bound: 1.0587202
time: 4.23 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0605684, upper bound: 1.0587198
time: 4.25 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6474874, 0.1031362, -1.6408455, 0.1300051, -1.6887193, 1.6481891
1: -17.9641285, -15.6763124, -17.9600792, -15.6552572, -1.9169397, 1.9030752
2: -6.5299435, -4.4793100, -6.5792809, -4.5007534, -1.7544985, 1.8130212
3: -13.9849815, -12.1281176, -14.0055599, -12.1412230, -1.5501137, 1.5785365
4: -5.5637589, -3.7327278, -5.5668731, -3.7035234, -1.8602355, 1.8341453
5: -7.0620365, -5.6056719, -7.0429173, -5.6024947, -1.2819145, 1.2664363
6: 8.2927990, 10.0501604, 8.3151035, 10.0787392, -1.6167626, 1.5456288
7: -14.0083485, -12.1327143, -14.0345783, -12.1279573, -1.4542203, 1.4887125
8: -6.0966654, -4.6709442, -6.0784225, -4.6339841, -1.1127276, 1.0618129
9: -10.8171368, -8.5148392, -10.8286991, -8.5229626, -2.0972948, 2.1075187

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0326150, upper bound: 1.0409052
time: 3.64 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0540302, upper bound: 1.0472817
time: 3.69 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6440014, 0.1040402, -1.6410147, 0.1308787, -1.6838431, 1.6582212
1: -17.9602852, -15.6609287, -17.9600925, -15.6518440, -1.9175153, 1.9330764
2: -6.5233088, -4.4851484, -6.5795999, -4.5005484, -1.7548571, 1.8110175
3: -13.9770088, -12.1308727, -14.0056801, -12.1403723, -1.5454326, 1.5848613
4: -5.6130128, -3.7180994, -5.5723267, -3.7033584, -1.9096544, 1.8542273
5: -7.0502768, -5.6142168, -7.0431242, -5.6016593, -1.2735100, 1.2736652
6: 8.2976637, 10.0353537, 8.3133726, 10.0787506, -1.6173348, 1.5473814
7: -14.0087109, -12.1323614, -14.0346870, -12.1278963, -1.4550905, 1.4892259
8: -6.0999498, -4.6585493, -6.0788412, -4.6320820, -1.1170261, 1.0769014
9: -10.8331833, -8.5093851, -10.8304892, -8.5228643, -2.1267881, 2.1073518

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0606059, upper bound: 1.0626115
time: 4.20 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0606059, upper bound: 1.0632249
time: 6.61 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6474874, 0.1031362, -1.6417774, 0.1253705, -1.6806149, 1.6497030
1: -17.9641285, -15.6763124, -17.9519863, -15.6310616, -1.9397492, 1.8865976
2: -6.5299435, -4.4793100, -6.5834241, -4.5007067, -1.7549362, 1.8182802
3: -13.9849815, -12.1281176, -14.0150003, -12.1182394, -1.5592084, 1.5772033
4: -5.5637589, -3.7327278, -5.6035395, -3.6893814, -1.8743775, 1.8708117
5: -7.0620365, -5.6056719, -7.0416508, -5.5886426, -1.2765081, 1.2410023
6: 8.2927990, 10.0501604, 8.3140850, 10.1027641, -1.6153622, 1.5193524
7: -14.0083485, -12.1327143, -14.0353584, -12.1245050, -1.4561045, 1.4888728
8: -6.0966654, -4.6709442, -6.0942678, -4.6341138, -1.1028399, 1.0690106
9: -10.8171368, -8.5148392, -10.8456373, -8.5268364, -2.0874786, 2.1247387

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0326891, upper bound: 1.0409571
time: 3.59 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0541817, upper bound: 1.0473903
time: 4.16 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6440014, 0.1040402, -1.6419480, 0.1262433, -1.6757398, 1.6598082
1: -17.9602852, -15.6609287, -17.9519997, -15.6275787, -1.9403272, 1.9166408
2: -6.5233088, -4.4851484, -6.5837455, -4.5005035, -1.7552958, 1.8162889
3: -13.9770088, -12.1308727, -14.0151215, -12.1173935, -1.5545273, 1.5835280
4: -5.6130128, -3.7180994, -5.6089945, -3.6892047, -1.9238081, 1.8908951
5: -7.0502768, -5.6142168, -7.0418663, -5.5878100, -1.2680666, 1.2482190
6: 8.2976637, 10.0353537, 8.3123579, 10.1027765, -1.6160028, 1.5211048
7: -14.0087109, -12.1323614, -14.0354652, -12.1244431, -1.4569764, 1.4893861
8: -6.0999498, -4.6585493, -6.0946827, -4.6322107, -1.1071613, 1.0842904
9: -10.8331833, -8.5093851, -10.8474274, -8.5267429, -2.1169767, 2.1245456

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0607497, upper bound: 1.0626806
time: 3.93 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0607497, upper bound: 1.0633189
time: 5.68 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6362944, 0.0979736, -1.6509105, 0.0977132, -1.6460600, 1.6616783
1: -17.8925724, -15.6427460, -17.9474869, -15.6737080, -1.7854776, 1.7923038
2: -6.5712786, -4.6014009, -6.5286012, -4.4970407, -1.7786489, 1.6475134
3: -13.9450569, -12.1577578, -13.9741182, -12.1296263, -1.5201216, 1.5265286
4: -5.5139275, -3.7014916, -5.5522261, -3.7351825, -1.7787449, 1.8507345
5: -7.0069103, -5.5881286, -7.0548182, -5.6016545, -1.2091048, 1.2782727
6: 8.3444939, 10.0807228, 8.2983789, 10.0525513, -1.4807000, 1.5957131
7: -14.0296717, -12.2535143, -14.0084400, -12.1545124, -1.4205861, 1.2974195
8: -6.0299654, -4.6201167, -6.0911760, -4.6709685, -1.0104507, 1.1205180
9: -10.7601833, -8.5254250, -10.8082333, -8.5152674, -2.0149188, 2.0631442

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0362067, upper bound: 1.0326166
time: 4.75 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0362067, upper bound: 1.0326148
time: 4.33 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6398058, 0.1235261, -1.6520731, 0.1031385, -1.6471381, 1.6809163
1: -17.9312572, -15.6599655, -17.9641266, -15.6691675, -1.7784655, 1.8870809
2: -6.5775747, -4.5219831, -6.5306892, -4.4790859, -1.8114896, 1.6779816
3: -14.0009422, -12.1415291, -13.9849882, -12.1295509, -1.5289350, 1.5497327
4: -5.5481653, -3.7063713, -5.5624719, -3.7323999, -1.8157654, 1.8561006
5: -7.0283108, -5.6028171, -7.0623093, -5.6016283, -1.2311196, 1.2836081
6: 8.3410339, 10.0768242, 8.2927876, 10.0546913, -1.5109797, 1.6132336
7: -14.0341940, -12.1480951, -14.0084648, -12.1311369, -1.4904900, 1.2905781
8: -6.0588017, -4.6344018, -6.1001935, -4.6709433, -1.0398564, 1.1142545
9: -10.8135128, -8.5234947, -10.8218365, -8.5148382, -2.0771680, 2.1019492

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0426581, upper bound: 1.0540298
time: 4.57 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0426581, upper bound: 1.0540302
time: 5.73 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6431553, 0.1216086, -1.6485811, 0.1040424, -1.6544042, 1.6715007
1: -17.9636688, -15.6874466, -17.9602833, -15.6537380, -1.9240966, 1.8810036
2: -6.5762124, -4.4960995, -6.5240536, -4.4849262, -1.8039465, 1.7572479
3: -14.0129337, -12.1494884, -13.9770184, -12.1323032, -1.5859594, 1.5337462
4: -5.5157537, -3.7187407, -5.6117249, -3.7177689, -1.7979848, 1.8929842
5: -7.0536461, -5.6105800, -7.0505390, -5.6101704, -1.2724624, 1.2632270
6: 8.3314953, 10.0934458, 8.2976522, 10.0398808, -1.5142283, 1.6284208
7: -14.0335865, -12.1285839, -14.0088320, -12.1307831, -1.4909256, 1.4525793
8: -6.0733161, -4.6517792, -6.1035018, -4.6585503, -1.0708702, 1.0969152
9: -10.8115711, -8.5287819, -10.8378792, -8.5093870, -2.0925064, 2.0963793

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0563528, upper bound: 1.0606076
time: 4.20 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0563528, upper bound: 1.0606055
time: 6.54 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6399492, 0.1225136, -1.6485811, 0.1040424, -1.6543078, 1.6822414
1: -17.9598312, -15.6711998, -17.9602833, -15.6537380, -1.9340706, 1.9119883
2: -6.5695758, -4.5019503, -6.5240536, -4.4849262, -1.8041630, 1.7545605
3: -14.0048733, -12.1522398, -13.9770184, -12.1323032, -1.5837045, 1.5400877
4: -5.5648775, -3.7041807, -5.6117249, -3.7177689, -1.8471086, 1.9075441
5: -7.0419526, -5.6186638, -7.0505390, -5.6101704, -1.2688832, 1.2702221
6: 8.3368349, 10.0786743, 8.2976522, 10.0398808, -1.5145950, 1.6172547
7: -14.0339508, -12.1282454, -14.0088320, -12.1307831, -1.4916162, 1.4541423
8: -6.0767317, -4.6393857, -6.1035018, -4.6585503, -1.0730314, 1.1130359
9: -10.8275604, -8.5233507, -10.8378792, -8.5093870, -2.1220012, 2.1211190

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0563528, upper bound: 1.0621322
time: 4.15 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0563528, upper bound: 1.0621305
time: 3.93 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6441225, 0.1169778, -1.6438589, 0.1161822, -1.6714864, 1.6652818
1: -17.9555759, -15.6633873, -17.9686317, -15.6681881, -1.8954706, 1.9212017
2: -6.5803542, -4.4960556, -6.5288963, -4.4839773, -1.8091431, 1.7582579
3: -14.0223732, -12.1265011, -13.9682255, -12.1428385, -1.5841966, 1.5446782
4: -5.5524216, -3.7047129, -5.5802670, -3.7302136, -1.8222079, 1.8755541
5: -7.0524278, -5.5967140, -7.0525498, -5.6114335, -1.2667682, 1.2803969
6: 8.3304405, 10.1175041, 8.2791271, 10.0093517, -1.4920616, 1.6667757
7: -14.0343666, -12.1251354, -14.0085640, -12.1355600, -1.4877148, 1.4553995
8: -6.0891891, -4.6519094, -6.0857420, -4.6530190, -1.0921817, 1.0822612
9: -10.8285151, -8.5326481, -10.8178778, -8.5051088, -2.1264467, 2.0792356

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_A2_B1_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0268153, upper bound: 1.0460766
time: 4.27 seconds

## Relational analysis of IS_A2_B1_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0489624, upper bound: 1.0544646
time: 3.59 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 13.67 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0472748, upper bound: 1.0401142
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0472748, upper bound: 1.0401138
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0542874, upper bound: 1.0605371
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0542874, upper bound: 1.0605371
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0672396
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0672402
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0684570
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0684570
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0401139, upper bound: 1.0542197
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0605354, upper bound: 1.0615938
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0753438
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0672400, upper bound: 1.0757883
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0401139, upper bound: 1.0542199
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0612062, upper bound: 1.0619329
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0678214, upper bound: 1.0756495
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0678214, upper bound: 1.0761132
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0393675, upper bound: 1.0268148
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0393675, upper bound: 1.0268150
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0477372, upper bound: 1.0489623
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0477372, upper bound: 1.0489617
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0605684, upper bound: 1.0565615
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0605684, upper bound: 1.0565620
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0605684, upper bound: 1.0587202
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0605684, upper bound: 1.0587198
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0326150, upper bound: 1.0409052
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0540302, upper bound: 1.0472817
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0606059, upper bound: 1.0626115
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0606059, upper bound: 1.0632249
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0326891, upper bound: 1.0409571
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0541817, upper bound: 1.0473903
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0607497, upper bound: 1.0626806
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0607497, upper bound: 1.0633189
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0362067, upper bound: 1.0326166
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0362067, upper bound: 1.0326148
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0426581, upper bound: 1.0540298
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0426581, upper bound: 1.0540302
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0563528, upper bound: 1.0606076
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0563528, upper bound: 1.0606055
IS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0563528, upper bound: 1.0621322
IS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0563528, upper bound: 1.0621305
IS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0268153, upper bound: 1.0460766
IS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.67
Output dim: 6, lower bound: -1.0489624, upper bound: 1.0544646
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0689530
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 6, lower bound: -1.0631797, upper bound: 1.0674225
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0691045
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 6, lower bound: -1.0409052, upper bound: 1.0212296
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 6, lower bound: -1.0472816, upper bound: 1.0435771
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 6, lower bound: -1.0626097, upper bound: 1.0506519
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 6, lower bound: -1.0626097, upper bound: 1.0526794
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 6, lower bound: -1.0631797, upper bound: 1.0563876
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0583663
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 6, lower bound: -1.0631797, upper bound: 1.0564494
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 13.67
Output dim: 6, lower bound: -1.0585349, upper bound: 1.0585298
Binary search (step 0): status=Status.UNKNOWN, k_low=3, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=1.614668846130371
rel_dist={6: [-1.1326113226070742, 1.1326113586240112]}

## Binary search (step 1) starts
Candidate k: 4, corresponding eps: 0.0156250


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2130
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2130

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8125517, upper bound: 0.8153173
time: 3.82 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8153154, upper bound: 0.8153159
time: 9.56 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 13.68 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 13.68
Output dim: 6, lower bound: -0.8125517, upper bound: 0.8153173
IS_A2, status: Status.UNKNOWN, split count: 1, time: 13.68
Output dim: 6, lower bound: -0.8153154, upper bound: 0.8153159

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -1.6506550, 0.1124235, -1.6561922, 0.1125274, -1.5001407, 1.5080438
1: -17.9605484, -15.6270342, -17.9640560, -15.6267805, -1.5740376, 1.5782840
2: -6.5349741, -4.4834495, -6.5349898, -4.4621940, -1.4710484, 1.4472051
3: -13.9778080, -12.1164856, -13.9784403, -12.1074400, -1.3269873, 1.2967229
4: -5.6252713, -3.7163892, -5.6369295, -3.7163720, -1.8411865, 1.8574867
5: -7.0521522, -5.5889454, -7.0538297, -5.5888457, -1.0651581, 1.0621344
6: 8.2744751, 10.0458050, 8.2564125, 10.0458202, -1.3519273, 1.3649936
7: -14.0097361, -12.1294889, -14.0097389, -12.1174345, -1.1625245, 1.1561034
8: -6.1093316, -4.6512423, -6.1156301, -4.6512346, -0.8946238, 0.8973209
9: -10.8447342, -8.5088797, -10.8449526, -8.5048056, -1.8542910, 1.8527431

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8089568
time: 3.80 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8153171
time: 3.79 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.6475005, 0.1262470, -1.6520326, 0.1124018, -1.4985662, 1.5265756
1: -17.9519997, -15.6137323, -17.9604244, -15.6270275, -1.6026764, 1.5923021
2: -6.5853863, -4.5002308, -6.5349836, -4.4760737, -1.5352578, 1.4556661
3: -14.0151386, -12.1148701, -13.9777927, -12.1096754, -1.3656096, 1.3342991
4: -5.6131539, -3.6883640, -5.6286044, -3.7163811, -1.8402424, 1.8906693
5: -7.0425291, -5.5800605, -7.0503912, -5.5889564, -1.0942030, 1.0603985
6: 8.3123417, 10.1139135, 8.2783871, 10.0458031, -1.3209300, 1.4194174
7: -14.0357504, -12.1219559, -14.0097370, -12.1223621, -1.1811223, 1.1480534
8: -6.1018839, -4.6322112, -6.1094365, -4.6512480, -0.8826356, 0.8913603
9: -10.8561497, -8.5267410, -10.8447075, -8.5121737, -1.8477702, 1.8509269

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8153156, upper bound: 0.8089563
time: 3.78 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8153156, upper bound: 0.8153161
time: 3.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 13.46 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.46
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8089568
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.46
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8153171
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.46
Output dim: 6, lower bound: -0.8153156, upper bound: 0.8089563
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.46
Output dim: 6, lower bound: -0.8153156, upper bound: 0.8153161

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -1.6506550, 0.1124235, -1.6506550, 0.1124235, -1.4989934, 1.4989934
1: -17.9605484, -15.6270342, -17.9605484, -15.6270342, -1.5659714, 1.5659711
2: -6.5349741, -4.4834495, -6.5349741, -4.4834495, -1.4471903, 1.4471903
3: -13.9778080, -12.1164856, -13.9778080, -12.1164856, -1.2840953, 1.2840958
4: -5.6252713, -3.7163892, -5.6252713, -3.7163892, -1.8410082, 1.8410082
5: -7.0521522, -5.5889454, -7.0521522, -5.5889454, -1.0555890, 1.0555890
6: 8.2744751, 10.0458050, 8.2744751, 10.0458050, -1.3519087, 1.3519087
7: -14.0097361, -12.1294889, -14.0097361, -12.1294889, -1.1561005, 1.1561005
8: -6.1093316, -4.6512423, -6.1093316, -4.6512423, -0.8946166, 0.8946166
9: -10.8447342, -8.5088797, -10.8447342, -8.5088797, -1.8526292, 1.8526297

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8044769, upper bound: 0.7949408
time: 4.05 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8052161, upper bound: 0.8011779
time: 5.86 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -1.6506550, 0.1124235, -1.6475005, 0.1262470, -1.5161834, 1.4991717
1: -17.9605484, -15.6270342, -17.9519997, -15.6137323, -1.5832372, 1.5625010
2: -6.5349741, -4.4834495, -6.5853863, -4.5002308, -1.4628019, 1.5175419
3: -13.9778080, -12.1164856, -14.0151386, -12.1148701, -1.2908068, 1.3250647
4: -5.6252713, -3.7163892, -5.6131539, -3.6883640, -1.8790531, 1.8432484
5: -7.0521522, -5.5889454, -7.0425291, -5.5800605, -1.0621996, 1.0479164
6: 8.2744751, 10.0458050, 8.3123417, 10.1139135, -1.4342155, 1.3105640
7: -14.0097361, -12.1294889, -14.0357504, -12.1219559, -1.1476970, 1.1812129
8: -6.1093316, -4.6512423, -6.1018839, -4.6322112, -0.8972647, 0.8767762
9: -10.8447342, -8.5088797, -10.8561497, -8.5267410, -1.8258533, 1.8552079

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8044769, upper bound: 0.8015877
time: 4.66 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8052161, upper bound: 0.8073649
time: 4.14 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.6475005, 0.1262470, -1.6506550, 0.1124235, -1.4991717, 1.5161834
1: -17.9519997, -15.6137323, -17.9605484, -15.6270342, -1.5625010, 1.5832372
2: -6.5853863, -4.5002308, -6.5349741, -4.4834495, -1.5175419, 1.4628024
3: -14.0151386, -12.1148701, -13.9778080, -12.1164856, -1.3250647, 1.2908063
4: -5.6131539, -3.6883640, -5.6252713, -3.7163892, -1.8432484, 1.8790531
5: -7.0425291, -5.5800605, -7.0521522, -5.5889454, -1.0479164, 1.0621995
6: 8.3123417, 10.1139135, 8.2744751, 10.0458050, -1.3105640, 1.4342155
7: -14.0357504, -12.1219559, -14.0097361, -12.1294889, -1.1812127, 1.1476972
8: -6.1018839, -4.6322112, -6.1093316, -4.6512423, -0.8767762, 0.8972646
9: -10.8561497, -8.5267410, -10.8447342, -8.5088797, -1.8552079, 1.8258529

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8067780, upper bound: 0.7949410
time: 7.01 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8073647, upper bound: 0.8011783
time: 6.31 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.6475005, 0.1262470, -1.6475005, 0.1262470, -1.5218244, 1.5218244
1: -17.9519997, -15.6137323, -17.9519997, -15.6137323, -1.6045084, 1.6045082
2: -6.5853863, -4.5002308, -6.5853863, -4.5002308, -1.4615407, 1.4615407
3: -14.0151386, -12.1148701, -14.0151386, -12.1148701, -1.3085542, 1.3085542
4: -5.6131539, -3.6883640, -5.6131539, -3.6883640, -1.8396626, 1.8396626
5: -7.0425291, -5.5800605, -7.0425291, -5.5800605, -1.0938368, 1.0938368
6: 8.3123417, 10.1139135, 8.3123417, 10.1139135, -1.3388152, 1.3388155
7: -14.0357504, -12.1219559, -14.0357504, -12.1219559, -1.1718147, 1.1718147
8: -6.1018839, -4.6322112, -6.1018839, -4.6322112, -0.8813893, 0.8813894
9: -10.8561497, -8.5267410, -10.8561497, -8.5267410, -1.8504500, 1.8504496

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8067782, upper bound: 0.7949414
time: 7.83 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8073649, upper bound: 0.8011781
time: 5.61 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 19.21 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.21
Output dim: 6, lower bound: -0.8044769, upper bound: 0.7949408
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.21
Output dim: 6, lower bound: -0.8052161, upper bound: 0.8011779
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.21
Output dim: 6, lower bound: -0.8044769, upper bound: 0.8015877
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.21
Output dim: 6, lower bound: -0.8052161, upper bound: 0.8073649
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 19.21
Output dim: 6, lower bound: -0.8067780, upper bound: 0.7949410
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 19.21
Output dim: 6, lower bound: -0.8073647, upper bound: 0.8011783
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 19.21
Output dim: 6, lower bound: -0.8067782, upper bound: 0.7949414
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 19.21
Output dim: 6, lower bound: -0.8073649, upper bound: 0.8011781

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6440289, 0.1170566, -1.6482308, 0.1124225, -1.4872265, 1.4952326
1: -17.9686394, -15.6648798, -17.9605503, -15.6416302, -1.5535336, 1.5210278
2: -6.5292134, -4.4837713, -6.5328522, -4.4835677, -1.4406943, 1.4443722
3: -13.9683418, -12.1419897, -13.9777975, -12.1258564, -1.2703619, 1.2653995
4: -5.5857277, -3.7300534, -5.6107421, -3.7176168, -1.8030415, 1.8166342
5: -7.0527544, -5.6105976, -7.0511942, -5.5977745, -1.0287926, 1.0215199
6: 8.2773838, 10.0093632, 8.2744865, 10.0319519, -1.3233404, 1.3125248
7: -14.0086727, -12.1354914, -14.0093460, -12.1316643, -1.1527622, 1.1497090
8: -6.0861721, -4.6511135, -6.1007776, -4.6512446, -0.8693271, 0.8828506
9: -10.8196745, -8.5050039, -10.8355474, -8.5088806, -1.8219304, 1.8420944

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8001028, upper bound: 0.7987835
time: 3.81 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8003393, upper bound: 0.7938902
time: 3.81 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6494200, 0.1124224, -1.4893079, 1.4983506
1: -17.9605503, -15.6407909, -17.9605503, -15.6300259, -1.5635581, 1.5404007
2: -6.5333314, -4.4837227, -6.5346189, -4.4835072, -1.4456182, 1.4470773
3: -13.9777908, -12.1190071, -13.9778023, -12.1171074, -1.2833476, 1.2684808
4: -5.6204309, -3.7172384, -5.6242318, -3.7165711, -1.8086805, 1.8368926
5: -7.0514894, -5.5967188, -7.0520096, -5.5907087, -1.0524638, 1.0153918
6: 8.2744904, 10.0354328, 8.2744780, 10.0435734, -1.3511276, 1.3083405
7: -14.0094528, -12.1319809, -14.0096779, -12.1300278, -1.1540751, 1.1528692
8: -6.1021070, -4.6512437, -6.1077771, -4.6512423, -0.8746753, 0.8930857
9: -10.8361607, -8.5088787, -10.8425074, -8.5088787, -1.8384519, 1.8511333

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8008062, upper bound: 0.8057623
time: 3.99 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8010869, upper bound: 0.8010866
time: 3.56 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6440289, 0.1170566, -1.6451349, 0.1262449, -1.5044165, 1.4953876
1: -17.9686394, -15.6648798, -17.9519997, -15.6285629, -1.5710857, 1.5175569
2: -6.5292134, -4.4837713, -6.5832639, -4.5003462, -1.4563060, 1.5147381
3: -13.9683418, -12.1419897, -14.0151272, -12.1242409, -1.2770720, 1.3063636
4: -5.5857277, -3.7300534, -5.5981541, -3.6895776, -1.8411112, 1.8188510
5: -7.0527544, -5.6105976, -7.0415726, -5.5888896, -1.0357909, 1.0138665
6: 8.2773838, 10.0093632, 8.3123531, 10.1005316, -1.4061313, 1.2711797
7: -14.0086727, -12.1354914, -14.0353594, -12.1241159, -1.1442676, 1.1748366
8: -6.0861721, -4.6511135, -6.0933781, -4.6322112, -0.8719757, 0.8649926
9: -10.8196745, -8.5050039, -10.8467379, -8.5267420, -1.7951517, 1.8442788

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7932718, upper bound: 0.7951595
time: 4.38 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7938512, upper bound: 0.7909634
time: 4.36 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6462853, 0.1262459, -1.5064950, 1.4985271
1: -17.9605503, -15.6407909, -17.9519997, -15.6167355, -1.5806603, 1.5383306
2: -6.5333314, -4.4837227, -6.5850306, -4.5002894, -1.4612298, 1.5173769
3: -13.9777908, -12.1190071, -14.0151358, -12.1154919, -1.2900586, 1.3090568
4: -5.6204309, -3.7172384, -5.6122589, -3.6885428, -1.8467236, 1.8391323
5: -7.0514894, -5.5967188, -7.0423880, -5.5818019, -1.0590019, 1.0077238
6: 8.2744904, 10.0354328, 8.3123465, 10.1115198, -1.4332724, 1.2686918
7: -14.0094528, -12.1319809, -14.0356884, -12.1224985, -1.1458473, 1.1779768
8: -6.1021070, -4.6512437, -6.1003351, -4.6322103, -0.8772327, 0.8752431
9: -10.8361607, -8.5088787, -10.8538885, -8.5267420, -1.8116736, 1.8536539

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7938624, upper bound: 0.8007143
time: 4.79 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7945636, upper bound: 0.7967204
time: 3.97 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6482308, 0.1124225, -1.4873099, 1.5124202
1: -17.9600925, -15.6518440, -17.9605503, -15.6416302, -1.5514627, 1.5391591
2: -6.5795999, -4.5005484, -6.5328522, -4.4835677, -1.5110440, 1.4599857
3: -14.0056801, -12.1403723, -13.9777975, -12.1258564, -1.3109217, 1.2721095
4: -5.5723267, -3.7033584, -5.6107421, -3.7176168, -1.8052588, 1.8547382
5: -7.0431242, -5.6016593, -7.0511942, -5.5977745, -1.0211725, 1.0290442
6: 8.3133726, 10.0787506, 8.2744865, 10.0319519, -1.2837000, 1.3961425
7: -14.0346870, -12.1278963, -14.0093460, -12.1316643, -1.1779172, 1.1413143
8: -6.0788412, -4.6320820, -6.1007776, -4.6512446, -0.8515139, 0.8854078
9: -10.8304892, -8.5228643, -10.8355474, -8.5088806, -1.8234353, 1.8153176

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7955057, upper bound: 0.7913486
time: 4.94 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7961506, upper bound: 0.7868379
time: 9.78 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6494200, 0.1124224, -1.4894538, 1.5155401
1: -17.9519997, -15.6275787, -17.9605503, -15.6300259, -1.5600901, 1.5585966
2: -6.5837455, -4.5005035, -6.5346189, -4.4835072, -1.5158830, 1.4626894
3: -14.0151215, -12.1173935, -13.9778023, -12.1171074, -1.3243084, 1.2751908
4: -5.6089945, -3.6892047, -5.6242318, -3.7165711, -1.8108969, 1.8749518
5: -7.0418663, -5.5878100, -7.0520096, -5.5907087, -1.0448022, 1.0229168
6: 8.3123579, 10.1027765, 8.2744780, 10.0435734, -1.3097830, 1.3918853
7: -14.0354652, -12.1244431, -14.0096779, -12.1300278, -1.1792259, 1.1446414
8: -6.0946827, -4.6322107, -6.1077771, -4.6512423, -0.8568141, 0.8957324
9: -10.8474274, -8.5267429, -10.8425074, -8.5088787, -1.8407578, 1.8243561

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7958822, upper bound: 0.7987801
time: 5.77 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7967185, upper bound: 0.7945634
time: 4.15 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6451349, 0.1262449, -1.5099621, 1.5180378
1: -17.9600925, -15.6518440, -17.9519997, -15.6285629, -1.5914688, 1.5597692
2: -6.5795999, -4.5005484, -6.5832639, -4.5003462, -1.4550524, 1.4587417
3: -14.0056801, -12.1403723, -14.0151272, -12.1242409, -1.2959690, 1.2906699
4: -5.5723267, -3.7033584, -5.5981541, -3.6895776, -1.8016701, 1.8152657
5: -7.0431242, -5.6016593, -7.0415726, -5.5888896, -1.0655618, 1.0596852
6: 8.3133726, 10.0787506, 8.3123531, 10.1005316, -1.3098679, 1.2994466
7: -14.0346870, -12.1278963, -14.0353594, -12.1241159, -1.1684270, 1.1654463
8: -6.0788412, -4.6320820, -6.0933781, -4.6322112, -0.8560611, 0.8695797
9: -10.8304892, -8.5228643, -10.8467379, -8.5267420, -1.8197556, 1.8399181

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7808090, upper bound: 0.7810363
time: 4.66 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7940376, upper bound: 0.7825448
time: 3.82 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6462853, 0.1262459, -1.5121031, 1.5211792
1: -17.9519997, -15.6275787, -17.9519997, -15.6167355, -1.6021595, 1.5782573
2: -6.5837455, -4.5005035, -6.5850306, -4.5002894, -1.4598837, 1.4613762
3: -14.0151215, -12.1173935, -14.0151358, -12.1154919, -1.3077116, 1.2945848
4: -5.6089945, -3.6892047, -5.6122589, -3.6885428, -1.8073139, 1.8355417
5: -7.0418663, -5.5878100, -7.0423880, -5.5818019, -1.0907145, 1.0518451
6: 8.3123579, 10.1027765, 8.3123465, 10.1115198, -1.3380232, 1.2949858
7: -14.0354652, -12.1244431, -14.0356884, -12.1224985, -1.1700027, 1.1687536
8: -6.0946827, -4.6322107, -6.1003351, -4.6322103, -0.8614075, 0.8798519
9: -10.8474274, -8.5267429, -10.8538885, -8.5267420, -1.8362761, 1.8489532

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7958822, upper bound: 0.7946615
time: 4.49 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7967185, upper bound: 0.7905817
time: 4.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 14.74 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.8001028, upper bound: 0.7987835
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.8003393, upper bound: 0.7938902
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.8008062, upper bound: 0.8057623
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.8010869, upper bound: 0.8010866
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7932718, upper bound: 0.7951595
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7938512, upper bound: 0.7909634
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7938624, upper bound: 0.8007143
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7945636, upper bound: 0.7967204
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7955057, upper bound: 0.7913486
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7961506, upper bound: 0.7868379
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7958822, upper bound: 0.7987801
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7967185, upper bound: 0.7945634
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7808090, upper bound: 0.7810363
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7940376, upper bound: 0.7825448
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7958822, upper bound: 0.7946615
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 14.74
Output dim: 6, lower bound: -0.7967185, upper bound: 0.7905817

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6434274, 0.1138980, -1.6506604, 0.1031381, -1.4715734, 1.4916034
1: -17.9685993, -15.6767483, -17.9641266, -15.6771240, -1.5164552, 1.5080104
2: -6.5280757, -4.4845252, -6.5294628, -4.4791555, -1.4401083, 1.4366255
3: -13.9679127, -12.1450491, -13.9849834, -12.1349697, -1.2583508, 1.2665243
4: -5.5660057, -3.7306409, -5.5540700, -3.7330978, -1.7558737, 1.7643161
5: -7.0520067, -5.6136150, -7.0617299, -5.6067338, -1.0156572, 1.0254854
6: 8.2836809, 10.0093212, 8.2927942, 10.0466728, -1.3138371, 1.2834601
7: -14.0082846, -12.1357317, -14.0082397, -12.1323967, -1.1511943, 1.1483748
8: -6.0846066, -4.6577721, -6.0953164, -4.6709437, -0.8448983, 0.8672205
9: -10.8131847, -8.5053768, -10.8165255, -8.5148392, -1.8102708, 1.8231502

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7856704, upper bound: 0.7744738
time: 4.14 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7871595, upper bound: 0.7852849
time: 3.74 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.6437480, 0.1147889, -1.6471783, 0.1040426, -1.4824200, 1.4876070
1: -17.9685688, -15.6707754, -17.9602833, -15.6617956, -1.5360770, 1.5117991
2: -6.5265274, -4.4841809, -6.5228281, -4.4849949, -1.4380431, 1.4372487
3: -13.9681158, -12.1451683, -13.9770145, -12.1377249, -1.2643800, 1.2624049
4: -5.5834680, -3.7302709, -5.6033239, -3.7184756, -1.7858543, 1.7645311
5: -7.0524244, -5.6152883, -7.0499811, -5.6152768, -1.0241911, 1.0173236
6: 8.2843056, 10.0093403, 8.2976589, 10.0318737, -1.3177629, 1.2867737
7: -14.0084753, -12.1355944, -14.0086060, -12.1320448, -1.1515756, 1.1493695
8: -6.0855827, -4.6530848, -6.0986071, -4.6585498, -0.8548123, 0.8759239
9: -10.8188448, -8.5051441, -10.8325691, -8.5093870, -1.8104830, 1.8449316

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7865642, upper bound: 0.7720406
time: 3.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7873107, upper bound: 0.7809152
time: 3.56 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.6444556, 0.1092628, -1.6518642, 0.1031392, -1.4736409, 1.4947543
1: -17.9605064, -15.6528034, -17.9641285, -15.6656036, -1.5265322, 1.5273702
2: -6.5321937, -4.4844775, -6.5312319, -4.4790940, -1.4450331, 1.4393172
3: -13.9773636, -12.1220675, -13.9849911, -12.1262150, -1.2713366, 1.2696061
4: -5.6007104, -3.7178721, -5.5675602, -3.7320740, -1.7615223, 1.7845993
5: -7.0507073, -5.5997291, -7.0625801, -5.5996628, -1.0393021, 1.0194318
6: 8.2807732, 10.0353899, 8.2927856, 10.0583086, -1.3416424, 1.2793016
7: -14.0090656, -12.1322222, -14.0085707, -12.1307592, -1.1524954, 1.1515298
8: -6.1005507, -4.6579003, -6.1023202, -4.6709428, -0.8502544, 0.8775342
9: -10.8296728, -8.5092487, -10.8234873, -8.5148373, -1.8268256, 1.8321919

Time for backsubstitution: 5.42 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7857619, upper bound: 0.7807160
time: 4.86 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7878466, upper bound: 0.7923049
time: 3.64 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.6447748, 0.1101545, -1.6483667, 0.1040433, -1.4845648, 1.4907436
1: -17.9604797, -15.6465807, -17.9602852, -15.6501122, -1.5460873, 1.5311584
2: -6.5306468, -4.4841318, -6.5245943, -4.4849353, -1.4429932, 1.4399562
3: -13.9775696, -12.1221848, -13.9770193, -12.1289730, -1.2773647, 1.2654901
4: -5.6181731, -3.7174735, -5.6168146, -3.7174397, -1.7914762, 1.7849154
5: -7.0511465, -5.6014066, -7.0507965, -5.6082072, -1.0478289, 1.0112386
6: 8.2814074, 10.0354099, 8.2976513, 10.0434952, -1.3455439, 1.2826872
7: -14.0092545, -12.1320848, -14.0089350, -12.1304073, -1.1528823, 1.1525269
8: -6.1015205, -4.6532121, -6.1056213, -4.6585503, -0.8603289, 0.8861527
9: -10.8353329, -8.5090151, -10.8395309, -8.5093851, -1.8270111, 1.8539205

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7866795, upper bound: 0.7783456
time: 4.08 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7880350, upper bound: 0.7880348
time: 3.52 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6434274, 0.1138980, -1.6473055, 0.1169797, -1.4887447, 1.4915466
1: -17.9685993, -15.6767483, -17.9555779, -15.6643257, -1.5336332, 1.5045033
2: -6.5280757, -4.4845252, -6.5798750, -4.4958992, -1.4556947, 1.5069909
3: -13.9679127, -12.1450491, -14.0223827, -12.1333542, -1.2650461, 1.3073206
4: -5.5660057, -3.7306409, -5.5415792, -3.7050788, -1.7939692, 1.7666163
5: -7.0520067, -5.6136150, -7.0521202, -5.5977988, -1.0229030, 1.0178738
6: 8.2836809, 10.0093212, 8.3304348, 10.1152563, -1.3966212, 1.2423763
7: -14.0082846, -12.1357317, -14.0342579, -12.1248035, -1.1427517, 1.1735172
8: -6.0846066, -4.6577721, -6.0878558, -4.6519094, -0.8475580, 0.8491801
9: -10.8131847, -8.5053768, -10.8278294, -8.5326490, -1.7835002, 1.8253970

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7783070, upper bound: 0.7679501
time: 4.32 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7814082, upper bound: 0.7827428
time: 4.63 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6437480, 0.1147889, -1.6440673, 0.1178833, -1.4996085, 1.4875517
1: -17.9685688, -15.6707754, -17.9517345, -15.6477852, -1.5532100, 1.5082932
2: -6.5265274, -4.4841809, -6.5732403, -4.5017495, -1.4536581, 1.5076170
3: -13.9681158, -12.1451683, -14.0143223, -12.1361084, -1.2710843, 1.3033257
4: -5.5834680, -3.7302709, -5.5907030, -3.6904469, -1.8239346, 1.7667999
5: -7.0524244, -5.6152883, -7.0403585, -5.6058865, -1.0314102, 1.0097132
6: 8.2843056, 10.0093403, 8.3357944, 10.1004543, -1.4005504, 1.2454243
7: -14.0084753, -12.1355944, -14.0346241, -12.1244650, -1.1431055, 1.1745090
8: -6.0855827, -4.6530848, -6.0912695, -4.6395159, -0.8574718, 0.8578137
9: -10.8188448, -8.5051441, -10.8438110, -8.5272207, -1.7837081, 1.8471723

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7792888, upper bound: 0.7662620
time: 4.09 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7818211, upper bound: 0.7786175
time: 3.82 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6444556, 0.1092628, -1.6484709, 0.1169793, -1.4908113, 1.4947181
1: -17.9605064, -15.6528034, -17.9555779, -15.6526470, -1.5431836, 1.5252650
2: -6.5321937, -4.4844775, -6.5816422, -4.4958386, -1.4606190, 1.5096149
3: -13.9773636, -12.1220675, -14.0223894, -12.1246033, -1.2780304, 1.3100023
4: -5.6007104, -3.7178721, -5.5556860, -3.7040634, -1.7995892, 1.7869229
5: -7.0507073, -5.5997291, -7.0529718, -5.5907044, -1.0460885, 1.0118053
6: 8.2807732, 10.0353899, 8.3304272, 10.1262579, -1.4237809, 1.2399149
7: -14.0090656, -12.1322222, -14.0345907, -12.1231937, -1.1443388, 1.1766522
8: -6.1005507, -4.6579003, -6.0948400, -4.6519089, -0.8528223, 0.8595264
9: -10.8296728, -8.5092487, -10.8349762, -8.5326481, -1.8000531, 1.8347740

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7783327, upper bound: 0.7731171
time: 4.37 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7820320, upper bound: 0.7887224
time: 4.40 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6447748, 0.1101545, -1.6452153, 0.1178836, -1.5017519, 1.4907222
1: -17.9604797, -15.6465807, -17.9517345, -15.6359491, -1.5627315, 1.5290554
2: -6.5306468, -4.4841318, -6.5750079, -4.5016909, -1.4586067, 1.5102539
3: -13.9775696, -12.1221848, -14.0143309, -12.1273575, -1.2840695, 1.3060131
4: -5.6181731, -3.7174735, -5.6048107, -3.6894197, -1.8295279, 1.7872081
5: -7.0511465, -5.6014066, -7.0411744, -5.5987949, -1.0545876, 1.0036134
6: 8.2814074, 10.0354099, 8.3357849, 10.1114407, -1.4276867, 1.2430334
7: -14.0092545, -12.1320848, -14.0349503, -12.1228514, -1.1446908, 1.1776464
8: -6.1015205, -4.6532121, -6.0982399, -4.6395168, -0.8628966, 0.8680460
9: -10.8353329, -8.5090151, -10.8509598, -8.5272207, -1.8002377, 1.8564973

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7793319, upper bound: 0.7714265
time: 3.99 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7824464, upper bound: 0.7846706
time: 3.60 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6404084, 0.1277260, -1.6506604, 0.1031381, -1.4716401, 1.5087833
1: -17.9600525, -15.6639261, -17.9641266, -15.6771240, -1.5143790, 1.5259795
2: -6.5784616, -4.5012875, -6.5294628, -4.4791555, -1.5104580, 1.4522514
3: -14.0052443, -12.1434364, -13.9849834, -12.1349697, -1.2988915, 1.2732306
4: -5.5526276, -3.7039599, -5.5540700, -3.7330978, -1.7581244, 1.8024306
5: -7.0423765, -5.6046705, -7.0617299, -5.6067338, -1.0080683, 1.0330158
6: 8.3195925, 10.0787106, 8.2927942, 10.0466728, -1.2742929, 1.3670764
7: -14.0342979, -12.1281214, -14.0082397, -12.1323967, -1.1763518, 1.1399982
8: -6.0773177, -4.6387358, -6.0953164, -4.6709437, -0.8271676, 0.8697793
9: -10.8240299, -8.5232182, -10.8165255, -8.5148392, -1.8117948, 1.7963758

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7802578, upper bound: 0.7662566
time: 4.07 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7834521, upper bound: 0.7789364
time: 7.72 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.6407306, 0.1286153, -1.6471783, 0.1040426, -1.4823408, 1.5047860
1: -17.9600220, -15.6573734, -17.9602833, -15.6617956, -1.5339980, 1.5303359
2: -6.5769157, -4.5009522, -6.5228281, -4.4849949, -1.5083933, 1.4528637
3: -14.0054483, -12.1435528, -13.9770145, -12.1377249, -1.3048134, 1.2691145
4: -5.5700741, -3.7035804, -5.6033239, -3.7184756, -1.7880707, 1.8026404
5: -7.0427933, -5.6063123, -7.0499811, -5.6152768, -1.0165322, 1.0247366
6: 8.3203859, 10.0787287, 8.2976589, 10.0318737, -1.2781882, 1.3703928
7: -14.0344896, -12.1279888, -14.0086060, -12.1320448, -1.1767335, 1.1409805
8: -6.0782671, -4.6340504, -6.0986071, -4.6585498, -0.8367552, 0.8784831
9: -10.8296738, -8.5229959, -10.8325691, -8.5093870, -1.8120012, 1.8181543

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7813157, upper bound: 0.7640495
time: 7.79 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7838529, upper bound: 0.7747495
time: 4.55 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.6413407, 0.1230908, -1.6518642, 0.1031392, -1.4737835, 1.5119367
1: -17.9519596, -15.6398764, -17.9641285, -15.6656036, -1.5230579, 1.5452344
2: -6.5826054, -4.5012426, -6.5312319, -4.4790940, -1.5152960, 1.4549413
3: -14.0146837, -12.1204519, -13.9849911, -12.1262150, -1.3122826, 1.2763128
4: -5.5892963, -3.6898441, -5.5675602, -3.7320740, -1.7637730, 1.8226666
5: -7.0410843, -5.5908141, -7.0625801, -5.5996628, -1.0316715, 1.0269632
6: 8.3185625, 10.1027317, 8.2927856, 10.0583086, -1.3003941, 1.3628454
7: -14.0350761, -12.1246672, -14.0085707, -12.1307592, -1.1776476, 1.1433275
8: -6.0931721, -4.6388655, -6.1023202, -4.6709428, -0.8324767, 0.8801821
9: -10.8409710, -8.5270891, -10.8234873, -8.5148373, -1.8291488, 1.8054175

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7803044, upper bound: 0.7731032
time: 4.42 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7842377, upper bound: 0.7866189
time: 4.51 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.6416652, 0.1239810, -1.6483667, 0.1040433, -1.4845543, 1.5079250
1: -17.9519291, -15.6329088, -17.9602852, -15.6501122, -1.5426092, 1.5497568
2: -6.5810599, -4.5009050, -6.5245943, -4.4849353, -1.5132570, 1.4555693
3: -14.0148878, -12.1205702, -13.9770193, -12.1289730, -1.3182020, 1.2721982
4: -5.6067410, -3.6894398, -5.6168146, -3.7174397, -1.7936935, 1.8229814
5: -7.0415239, -5.5924597, -7.0507965, -5.6082072, -1.0401273, 1.0186520
6: 8.3193645, 10.1027517, 8.2976513, 10.0434952, -1.3042650, 1.3662305
7: -14.0352659, -12.1245375, -14.0089350, -12.1304073, -1.1780350, 1.1443069
8: -6.0941162, -4.6341805, -6.1056213, -4.6585503, -0.8422480, 0.8888016
9: -10.8466120, -8.5268688, -10.8395309, -8.5093851, -1.8293304, 1.8271427

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7813634, upper bound: 0.7709109
time: 3.92 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7846686, upper bound: 0.7824484
time: 3.96 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6387310, 0.1189513, -1.6406646, 0.0941229, -1.4806414, 1.5105567
1: -17.9266758, -15.6622066, -17.8844891, -15.6140871, -1.4379392, 1.4298065
2: -6.5752382, -4.5395837, -6.5751219, -4.6009893, -1.3404374, 1.3991957
3: -13.9817238, -12.1405354, -13.9546452, -12.1407766, -1.2608123, 1.2320251
4: -5.5517149, -3.7095306, -5.5452809, -3.6874547, -1.7496300, 1.7275290
5: -7.0290956, -5.6017437, -7.0049849, -5.5742555, -1.0459445, 1.0039322
6: 8.3246460, 10.0739851, 8.3416004, 10.1021891, -1.2735476, 1.2327812
7: -14.0346308, -12.1779299, -14.0304832, -12.2499971, -1.0111246, 1.0599325
8: -6.0595145, -4.6321392, -6.0444512, -4.6183372, -0.8492016, 0.8172982
9: -10.8009739, -8.5236807, -10.7772970, -8.5291557, -1.7651744, 1.7471485

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7730564, upper bound: 0.7687034
time: 6.71 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7713726, upper bound: 0.7697234
time: 4.74 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6408453, 0.1298716, -1.6441005, 0.1197652, -1.5003586, 1.5155873
1: -17.9555855, -15.6526070, -17.9231758, -15.6332541, -1.5604839, 1.4284778
2: -6.5793219, -4.5039287, -6.5814900, -4.5215812, -1.3669429, 1.4543052
3: -14.0049629, -12.1404219, -14.0105095, -12.1245461, -1.2947526, 1.2382679
4: -5.5685654, -3.7038207, -5.5794353, -3.6924045, -1.7967253, 1.7509384
5: -7.0407376, -5.6017084, -7.0267625, -5.5892134, -1.0634904, 1.0272989
6: 8.3175488, 10.0784407, 8.3383312, 10.0986223, -1.3035207, 1.2673869
7: -14.0346260, -12.1310444, -14.0349712, -12.1442785, -0.9995687, 1.1643553
8: -6.0757437, -4.6321449, -6.0736012, -4.6326237, -0.8522949, 0.8509957
9: -10.8280411, -8.5229511, -10.8314724, -8.5272636, -1.8195438, 1.8073196

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7879713, upper bound: 0.7719247
time: 4.32 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7838530, upper bound: 0.7723402
time: 3.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6413407, 0.1230908, -1.6484709, 0.1169793, -1.4964156, 1.5173626
1: -17.9519596, -15.6398764, -17.9555779, -15.6526470, -1.5649600, 1.5650222
2: -6.5826054, -4.5012426, -6.5816422, -4.4958386, -1.4593067, 1.4536328
3: -14.0146837, -12.1204519, -14.0223894, -12.1246033, -1.2957635, 1.2956209
4: -5.5892963, -3.6898441, -5.5556860, -3.7040634, -1.7601767, 1.7832789
5: -7.0410843, -5.5908141, -7.0529718, -5.5907044, -1.0777378, 1.0558379
6: 8.3185625, 10.1027317, 8.3304272, 10.1262579, -1.3283911, 1.2656198
7: -14.0350761, -12.1246672, -14.0345907, -12.1231937, -1.1684976, 1.1674552
8: -6.0931721, -4.6388655, -6.0948400, -4.6519089, -0.8369912, 0.8639169
9: -10.8409710, -8.5270891, -10.8349762, -8.5326481, -1.8246422, 1.8299875

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7803044, upper bound: 0.7684073
time: 8.64 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7842377, upper bound: 0.7827127
time: 4.24 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6416652, 0.1239810, -1.6452153, 0.1178836, -1.5072036, 1.5133653
1: -17.9519291, -15.6329088, -17.9517345, -15.6359491, -1.5843816, 1.5694766
2: -6.5810599, -4.5009050, -6.5750079, -4.5016909, -1.4572902, 1.4542637
3: -14.0148878, -12.1205702, -14.0143309, -12.1273575, -1.3016582, 1.2916098
4: -5.6067410, -3.6894398, -5.6048107, -3.6894197, -1.7900991, 1.7835908
5: -7.0415239, -5.5924597, -7.0411744, -5.5987949, -1.0862143, 1.0475761
6: 8.3193645, 10.1027517, 8.3357849, 10.1114407, -1.3324270, 1.2690387
7: -14.0352659, -12.1245375, -14.0349503, -12.1228514, -1.1688480, 1.1684308
8: -6.0941162, -4.6341805, -6.0982399, -4.6395168, -0.8468156, 0.8725426
9: -10.8466120, -8.5268688, -10.8509598, -8.5272207, -1.8248291, 1.8517375

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7813634, upper bound: 0.7664520
time: 6.38 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7846686, upper bound: 0.7785803
time: 4.47 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 16.64 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7856704, upper bound: 0.7744738
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7871595, upper bound: 0.7852849
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7865642, upper bound: 0.7720406
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7873107, upper bound: 0.7809152
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7857619, upper bound: 0.7807160
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7878466, upper bound: 0.7923049
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7866795, upper bound: 0.7783456
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7880350, upper bound: 0.7880348
IS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7783070, upper bound: 0.7679501
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7814082, upper bound: 0.7827428
IS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7792888, upper bound: 0.7662620
IS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7818211, upper bound: 0.7786175
IS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7783327, upper bound: 0.7731171
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7820320, upper bound: 0.7887224
IS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7793319, upper bound: 0.7714265
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7824464, upper bound: 0.7846706
IS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7802578, upper bound: 0.7662566
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7834521, upper bound: 0.7789364
IS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7813157, upper bound: 0.7640495
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7838529, upper bound: 0.7747495
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7803044, upper bound: 0.7731032
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7842377, upper bound: 0.7866189
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7813634, upper bound: 0.7709109
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7846686, upper bound: 0.7824484
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7730564, upper bound: 0.7687034
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7713726, upper bound: 0.7697234
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7879713, upper bound: 0.7719247
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7838530, upper bound: 0.7723402
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7803044, upper bound: 0.7684073
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7842377, upper bound: 0.7827127
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7813634, upper bound: 0.7664520
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.64
Output dim: 6, lower bound: -0.7846686, upper bound: 0.7785803

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6388890, 0.0821395, -1.6481624, 0.0913207, -1.4639821, 1.4621692
1: -17.9010887, -15.6673241, -17.9279461, -15.6870499, -1.3876195, 1.3505797
2: -6.5196543, -4.5900755, -6.5250335, -4.5192914, -1.3796835, 1.3210611
3: -13.9075336, -12.1615915, -13.9607658, -12.1351318, -1.1992593, 1.2290492
4: -5.5111122, -3.7293634, -5.5314393, -3.7392626, -1.6683006, 1.7110491
5: -7.0155115, -5.6001983, -7.0456572, -5.6067920, -0.9615448, 1.0083578
6: 8.3159895, 10.0109282, 8.3052416, 10.0420094, -1.2497416, 1.2491198
7: -14.0022182, -12.2630186, -14.0081825, -12.1839695, -1.0466828, 0.9915838
8: -6.0348949, -4.6439075, -6.0754595, -4.6710019, -0.7935029, 0.8588978
9: -10.7476864, -8.5080528, -10.7873535, -8.5158052, -1.7171087, 1.7679453

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7743129, upper bound: 0.7657671
time: 4.10 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7749489, upper bound: 0.7667032
time: 4.03 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6423539, 0.1074346, -1.6504761, 0.1021923, -1.4691911, 1.4821010
1: -17.9397736, -15.6813087, -17.9597111, -15.6778297, -1.3773909, 1.4765375
2: -6.5263615, -4.5053501, -6.5291710, -4.4824915, -1.4357934, 1.3497376
3: -13.9633522, -12.1453571, -13.9842815, -12.1350155, -1.2027183, 1.2653556
4: -5.5475607, -3.7334669, -5.5504551, -3.7335672, -1.6942978, 1.7592254
5: -7.0374742, -5.6139207, -7.0594172, -5.6067820, -0.9855173, 1.0234518
6: 8.3089733, 10.0074730, 8.2967815, 10.0463715, -1.2869182, 1.2771311
7: -14.0078735, -12.1558695, -14.0081778, -12.1355581, -1.1500354, 0.9813128
8: -6.0644770, -4.6581898, -6.0921192, -4.6710081, -0.8230922, 0.8635352
9: -10.7982569, -8.5059490, -10.8141804, -8.5149403, -1.7780790, 1.8226938

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7760745, upper bound: 0.7758004
time: 3.80 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7766439, upper bound: 0.7773962
time: 3.62 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6392320, 0.0826666, -1.6447604, 0.0920660, -1.4754753, 1.4578538
1: -17.9010601, -15.6540794, -17.9268627, -15.6723633, -1.3919554, 1.3604662
2: -6.5184250, -4.5896640, -6.5183969, -4.5250416, -1.3785453, 1.3216887
3: -13.9078083, -12.1617069, -13.9528704, -12.1378860, -1.2046766, 1.2254252
4: -5.5288429, -3.7289598, -5.5817490, -3.7246513, -1.6986771, 1.7122855
5: -7.0159974, -5.6003036, -7.0356169, -5.6153336, -0.9704707, 1.0001132
6: 8.3162556, 10.0109539, 8.3106194, 10.0272884, -1.2515798, 1.2566638
7: -14.0025902, -12.2629032, -14.0085449, -12.1836500, -1.0472510, 0.9922256
8: -6.0359254, -4.6392202, -6.0787020, -4.6586089, -0.7986712, 0.8693919
9: -10.7526808, -8.5077686, -10.8038044, -8.5102825, -1.7174587, 1.7927561

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7752398, upper bound: 0.7630020
time: 3.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7758890, upper bound: 0.7642236
time: 4.11 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6426761, 0.1083293, -1.6470006, 0.1030400, -1.4799552, 1.4784775
1: -17.9397430, -15.6755199, -17.9557800, -15.6625576, -1.3711419, 1.4803827
2: -6.5248151, -4.5050087, -6.5225353, -4.4882178, -1.4337292, 1.3501582
3: -13.9635563, -12.1454744, -13.9763079, -12.1377707, -1.2074189, 1.2612309
4: -5.5647526, -3.7330832, -5.5995731, -3.7189379, -1.7237597, 1.7593040
5: -7.0379162, -5.6155963, -7.0475197, -5.6153231, -0.9934831, 1.0152261
6: 8.3094482, 10.0074902, 8.3014059, 10.0315733, -1.2860036, 1.2806959
7: -14.0080643, -12.1557331, -14.0085430, -12.1352062, -1.1504200, 0.9823661
8: -6.0654335, -4.6535034, -6.0954161, -4.6586146, -0.8327997, 0.8721297
9: -10.8039408, -8.5057096, -10.8301659, -8.5094757, -1.7784281, 1.8388796

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7762672, upper bound: 0.7714616
time: 3.95 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7768217, upper bound: 0.7730680
time: 3.64 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6401160, 0.0774870, -1.6493587, 0.0913221, -1.4658632, 1.4652829
1: -17.8929977, -15.6432495, -17.9279461, -15.6758327, -1.3979979, 1.3704414
2: -6.5237074, -4.5900249, -6.5267782, -4.5192332, -1.3843055, 1.3237410
3: -13.9169846, -12.1386003, -13.9607706, -12.1263771, -1.2122207, 1.2321525
4: -5.5458159, -3.7167485, -5.5449314, -3.7381520, -1.6736860, 1.7329507
5: -7.0137281, -5.5863166, -7.0464945, -5.5997200, -0.9853067, 1.0023463
6: 8.3130064, 10.0366364, 8.3052330, 10.0536766, -1.2774644, 1.2449687
7: -14.0029964, -12.2595444, -14.0085115, -12.1823387, -1.0478036, 0.9946229
8: -6.0507069, -4.6440392, -6.0824814, -4.6710024, -0.7988183, 0.8693641
9: -10.7620821, -8.5118952, -10.7942390, -8.5158052, -1.7328129, 1.7764659

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7794602, upper bound: 0.7806615
time: 4.23 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7794602, upper bound: 0.7807185
time: 4.54 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6433829, 0.1027955, -1.6516807, 0.1021930, -1.4712439, 1.4852457
1: -17.9316826, -15.6573734, -17.9597111, -15.6663284, -1.3880081, 1.4952469
2: -6.5304585, -4.5053039, -6.5309386, -4.4824309, -1.4406967, 1.3523865
3: -13.9728022, -12.1223736, -13.9842863, -12.1262627, -1.2157011, 1.2684422
4: -5.5822649, -3.7207184, -5.5639458, -3.7325387, -1.6998634, 1.7797718
5: -7.0360308, -5.6000376, -7.0602646, -5.5997100, -1.0091441, 1.0174000
6: 8.3060532, 10.0335484, 8.2967701, 10.0580091, -1.3147922, 1.2727509
7: -14.0086546, -12.1523962, -14.0085068, -12.1339226, -1.1513326, 0.9842511
8: -6.0802894, -4.6583195, -6.0991187, -4.6710072, -0.8283145, 0.8738848
9: -10.8144703, -8.5098171, -10.8211317, -8.5149393, -1.7941895, 1.8317733

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7808013, upper bound: 0.7916357
time: 3.53 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7808013, upper bound: 0.7923075
time: 3.49 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6404576, 0.0780151, -1.6459407, 0.0920665, -1.4774489, 1.4609504
1: -17.8929710, -15.6299734, -17.9268627, -15.6609697, -1.4023113, 1.3802631
2: -6.5224795, -4.5896163, -6.5201430, -4.5249810, -1.3831663, 1.3243561
3: -13.9172602, -12.1387177, -13.9528790, -12.1291304, -1.2176352, 1.2285297
4: -5.5635476, -3.7163138, -5.5952387, -3.7235270, -1.7040291, 1.7342219
5: -7.0142283, -5.5864234, -7.0364399, -5.6082668, -0.9943109, 0.9940315
6: 8.3132801, 10.0366621, 8.3106098, 10.0389366, -1.2792459, 1.2526300
7: -14.0033712, -12.2594299, -14.0088758, -12.1820221, -1.0483720, 0.9952645
8: -6.0517321, -4.6393490, -6.0857563, -4.6586080, -0.8041277, 0.8797588
9: -10.7670794, -8.5116167, -10.8106909, -8.5102816, -1.7331305, 1.8012180

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7803615, upper bound: 0.7782257
time: 4.15 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7803615, upper bound: 0.7783475
time: 3.90 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6437062, 0.1036896, -1.6481900, 0.1030405, -1.4820781, 1.4816227
1: -17.9316559, -15.6513262, -17.9557800, -15.6508970, -1.3817127, 1.4990876
2: -6.5289111, -4.5049601, -6.5243020, -4.4881577, -1.4386106, 1.3527946
3: -13.9730072, -12.1224918, -13.9763145, -12.1290188, -1.2204018, 1.2643170
4: -5.5994587, -3.7203038, -5.6130619, -3.7178955, -1.7293262, 1.7799544
5: -7.0364866, -5.6017156, -7.0483131, -5.6082549, -1.0171635, 1.0091418
6: 8.3065367, 10.0335655, 8.3013964, 10.0431948, -1.3138666, 1.2765706
7: -14.0088425, -12.1522570, -14.0088730, -12.1335678, -1.1517224, 0.9853861
8: -6.0812421, -4.6536331, -6.1024256, -4.6586146, -0.8382461, 0.8824072
9: -10.8201590, -8.5095787, -10.8371172, -8.5094757, -1.7947083, 1.8478603

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7809150, upper bound: 0.7873106
time: 3.71 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7809150, upper bound: 0.7880372
time: 3.72 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6388890, 0.0821395, -1.6449425, 0.1051560, -1.4811935, 1.4619813
1: -17.9010887, -15.6673241, -17.9193974, -15.6742859, -1.4042463, 1.3470371
2: -6.5196543, -4.5900755, -6.5754271, -4.5349083, -1.3969092, 1.3913059
3: -13.9075336, -12.1615915, -13.9983425, -12.1335182, -1.2059546, 1.2702041
4: -5.5111122, -3.7293634, -5.5199137, -3.7112238, -1.7065048, 1.7133689
5: -7.0155115, -5.6001983, -7.0361314, -5.5978847, -0.9687574, 1.0011253
6: 8.3159895, 10.0109282, 8.3418131, 10.1104317, -1.3323822, 1.2086866
7: -14.0022182, -12.2630186, -14.0342035, -12.1748791, -1.0380714, 1.0167348
8: -6.0348949, -4.6439075, -6.0681248, -4.6519690, -0.7961700, 0.8409004
9: -10.7476864, -8.5080528, -10.7977238, -8.5335636, -1.6904712, 1.7686749

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7667408, upper bound: 0.7589413
time: 4.24 seconds

## Relational analysis of IS_A1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7676548, upper bound: 0.7594594
time: 3.96 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6423539, 0.1074346, -1.6471317, 0.1160247, -1.4863319, 1.4820323
1: -17.9397736, -15.6813087, -17.9511604, -15.6650391, -1.3954675, 1.4729984
2: -6.5263615, -4.5053501, -6.5795841, -4.4993672, -1.4511662, 1.4215221
3: -13.9633522, -12.1453571, -14.0216694, -12.1334000, -1.2095680, 1.3062229
4: -5.5475607, -3.7334669, -5.5379529, -3.7055459, -1.7323875, 1.7615108
5: -7.0374742, -5.6139207, -7.0498047, -5.5978513, -0.9929132, 1.0158490
6: 8.3089733, 10.0074730, 8.3345156, 10.1149454, -1.3705873, 1.2361450
7: -14.0078735, -12.1558695, -14.0341988, -12.1279593, -1.1416037, 1.0053618
8: -6.0644770, -4.6581898, -6.0847392, -4.6519728, -0.8264655, 0.8454412
9: -10.7982569, -8.5059490, -10.8254309, -8.5327435, -1.7512856, 1.8256783

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7693355, upper bound: 0.7723244
time: 5.31 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7707454, upper bound: 0.7740954
time: 3.92 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6392320, 0.0826666, -1.6417830, 0.1058942, -1.4926987, 1.4576387
1: -17.9010601, -15.6540794, -17.9183159, -15.6583824, -1.4085574, 1.3569264
2: -6.5184250, -4.5896640, -6.5687866, -4.5407057, -1.3958101, 1.3919330
3: -13.9078083, -12.1617069, -13.9903603, -12.1362724, -1.2113824, 1.2666969
4: -5.5288429, -3.7289598, -5.5700932, -3.6966028, -1.7368641, 1.7145739
5: -7.0159974, -5.6003036, -7.0260754, -5.6059675, -0.9776573, 0.9928818
6: 8.3162556, 10.0109539, 8.3476000, 10.0957069, -1.3342195, 1.2159076
7: -14.0025902, -12.2629032, -14.0345688, -12.1745653, -1.0386295, 1.0173745
8: -6.0359254, -4.6392202, -6.0715961, -4.6395750, -0.8013378, 0.8514125
9: -10.7526808, -8.5077686, -10.8141022, -8.5280685, -1.6908159, 1.7934771

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B2_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7677953, upper bound: 0.7572398
time: 4.37 seconds

## Relational analysis of IS_A1_B2_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7686620, upper bound: 0.7577338
time: 5.27 seconds

## BFS IS instance: IS_A1_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6426761, 0.1083293, -1.6439000, 0.1168711, -1.4971137, 1.4784374
1: -17.9397430, -15.6755199, -17.9472275, -15.6485538, -1.3891892, 1.4768460
2: -6.5248151, -4.5050087, -6.5729499, -4.5051289, -1.4491196, 1.4219408
3: -13.9635563, -12.1454744, -14.0136051, -12.1361551, -1.2142825, 1.3021188
4: -5.5647526, -3.7330832, -5.5869393, -3.6909080, -1.7618380, 1.7615252
5: -7.0379162, -5.6155963, -7.0378685, -5.6059341, -1.0009029, 1.0076233
6: 8.3094482, 10.0074902, 8.3396406, 10.1001434, -1.3696752, 1.2394400
7: -14.0080643, -12.1557331, -14.0345631, -12.1276150, -1.1419647, 1.0064118
8: -6.0654335, -4.6535034, -6.0881624, -4.6395807, -0.8361719, 0.8540286
9: -10.8039408, -8.5057096, -10.8413544, -8.5273056, -1.7516313, 1.8418503

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B2_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7697537, upper bound: 0.7681304
time: 4.21 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7711624, upper bound: 0.7699566
time: 3.81 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6401160, 0.0774870, -1.6461142, 0.1051553, -1.4830713, 1.4651175
1: -17.8929977, -15.6432495, -17.9193974, -15.6629162, -1.4141755, 1.3683029
2: -6.5237074, -4.5900249, -6.5771680, -4.5348487, -1.4015322, 1.3939261
3: -13.9169846, -12.1386003, -13.9983482, -12.1247616, -1.2189164, 1.2729187
4: -5.5458159, -3.7167485, -5.5340219, -3.7101243, -1.7118587, 1.7352953
5: -7.0137281, -5.5863166, -7.0369296, -5.5907893, -0.9920595, 0.9950978
6: 8.3130064, 10.0366364, 8.3418036, 10.1214695, -1.3594637, 1.2062409
7: -14.0029964, -12.2595444, -14.0345345, -12.1732731, -1.0393872, 1.0197530
8: -6.0507069, -4.6440392, -6.0751109, -4.6519675, -0.8013929, 0.8514087
9: -10.7620821, -8.5118952, -10.8048000, -8.5335627, -1.7061749, 1.7775207

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A2_B1_A1_B1

### Relational analysis result of IS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7715192, upper bound: 0.7730563
time: 4.10 seconds

## Relational analysis of IS_A1_B2_A2_B1_A1_B2

### Relational analysis result of IS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7715192, upper bound: 0.7731167
time: 4.70 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6433829, 0.1027955, -1.6482953, 0.1160235, -1.4883819, 1.4851975
1: -17.9316826, -15.6573734, -17.9511642, -15.6533813, -1.4055219, 1.4931116
2: -6.5304585, -4.5053039, -6.5813513, -4.4993081, -1.4560699, 1.4241247
3: -13.9728022, -12.1223736, -14.0216751, -12.1246490, -1.2225513, 1.3089085
4: -5.5822649, -3.7207184, -5.5520592, -3.7045255, -1.7379255, 1.7820802
5: -7.0360308, -5.6000376, -7.0506520, -5.5907550, -1.0160799, 1.0097823
6: 8.3060532, 10.0335484, 8.3345070, 10.1259470, -1.3978100, 1.2334611
7: -14.0086546, -12.1523962, -14.0345306, -12.1263447, -1.1431823, 1.0082796
8: -6.0802894, -4.6583195, -6.0917206, -4.6519718, -0.8315948, 0.8558451
9: -10.8144703, -8.5098171, -10.8325720, -8.5327435, -1.7673950, 1.8351007

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7743341, upper bound: 0.7879716
time: 4.28 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7743341, upper bound: 0.7887252
time: 3.87 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6404576, 0.0780151, -1.6429368, 0.1058938, -1.4946709, 1.4607573
1: -17.8929710, -15.6299734, -17.9183140, -15.6468382, -1.4184632, 1.3781269
2: -6.5224795, -4.5896163, -6.5705318, -4.5406461, -1.4004307, 1.3945427
3: -13.9172602, -12.1387177, -13.9903688, -12.1275167, -1.2243414, 1.2694163
4: -5.5635476, -3.7163138, -5.5842013, -3.6954889, -1.7421875, 1.7365355
5: -7.0142283, -5.5864234, -7.0268612, -5.5988731, -1.0010374, 0.9867842
6: 8.3132801, 10.0366621, 8.3475924, 10.1067286, -1.3612437, 1.2135687
7: -14.0033712, -12.2594299, -14.0348969, -12.1729603, -1.0399377, 1.0203927
8: -6.0517321, -4.6393490, -6.0785694, -4.6395741, -0.8067021, 0.8618389
9: -10.7670794, -8.5116167, -10.8211832, -8.5280657, -1.7064886, 1.8022776

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A2_B2_A1_B1

### Relational analysis result of IS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7725058, upper bound: 0.7713727
time: 4.21 seconds

## Relational analysis of IS_A1_B2_A2_B2_A1_B2

### Relational analysis result of IS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7725058, upper bound: 0.7714268
time: 4.64 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6437062, 0.1036896, -1.6450486, 0.1168718, -1.4992337, 1.4816031
1: -17.9316559, -15.6513262, -17.9472275, -15.6367378, -1.3992155, 1.4969547
2: -6.5289111, -4.5049601, -6.5747156, -4.5050697, -1.4540000, 1.4245324
3: -13.9730072, -12.1224918, -14.0136147, -12.1274042, -1.2272644, 1.3048091
4: -5.5994587, -3.7203038, -5.6010466, -3.6898742, -1.7673750, 1.7822003
5: -7.0364866, -5.6017156, -7.0386791, -5.5988431, -1.0241237, 1.0015252
6: 8.3065367, 10.0335655, 8.3396320, 10.1111317, -1.3968878, 1.2370095
7: -14.0088425, -12.1522570, -14.0348940, -12.1260033, -1.1435349, 1.0094118
8: -6.0812421, -4.6536331, -6.0951281, -4.6395798, -0.8415257, 0.8643095
9: -10.8201590, -8.5095787, -10.8484964, -8.5273056, -1.7679129, 1.8511572

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7747492, upper bound: 0.7838551
time: 4.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7747492, upper bound: 0.7846709
time: 3.95 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6358458, 0.0959520, -1.6481624, 0.0913207, -1.4637952, 1.4794269
1: -17.8925400, -15.6545353, -17.9279461, -15.6870499, -1.3855820, 1.3682649
2: -6.5703168, -4.6019506, -6.5250335, -4.5192914, -1.4526143, 1.3423686
3: -13.9446878, -12.1599722, -13.9607658, -12.1351318, -1.2408352, 1.2362413
4: -5.4994912, -3.7019687, -5.5314393, -3.7392626, -1.6725111, 1.7494631
5: -7.0063238, -5.5910110, -7.0456572, -5.6067920, -0.9545870, 1.0159734
6: 8.3491993, 10.0806894, 8.3052416, 10.0420094, -1.2126813, 1.3332486
7: -14.0293112, -12.2536488, -14.0081825, -12.1839695, -1.0707269, 0.9832964
8: -6.0288482, -4.6248693, -6.0754595, -4.6710019, -0.7757093, 0.8622322
9: -10.7557402, -8.5257349, -10.7873535, -8.5158052, -1.7150164, 1.7409277

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7680953, upper bound: 0.7570455
time: 4.81 seconds

## Relational analysis of IS_A2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7687321, upper bound: 0.7587523
time: 4.69 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6393664, 0.1212402, -1.6504761, 0.1021923, -1.4691858, 1.4991832
1: -17.9312267, -15.6685200, -17.9597111, -15.6778297, -1.3754554, 1.4947772
2: -6.5767560, -4.5225182, -6.5291710, -4.4824915, -1.5060954, 1.3689146
3: -14.0006256, -12.1437416, -13.9842815, -12.1350155, -1.2437153, 1.2720752
4: -5.5342245, -3.7068272, -5.5504551, -3.7335672, -1.6968174, 1.7973914
5: -7.0277410, -5.6049933, -7.0594172, -5.6067820, -0.9782398, 1.0309672
6: 8.3453932, 10.0767956, 8.2967815, 10.0463715, -1.2476678, 1.3607531
7: -14.0339127, -12.1482553, -14.0081778, -12.1355581, -1.1752529, 0.9719981
8: -6.0577106, -4.6391544, -6.0921192, -4.6710081, -0.8053987, 0.8661418
9: -10.8088264, -8.5237579, -10.8141804, -8.5149403, -1.7791519, 1.7958698

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7700981, upper bound: 0.7688013
time: 4.72 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7717527, upper bound: 0.7714507
time: 4.34 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6361830, 0.0964706, -1.6447604, 0.0920660, -1.4751701, 1.4751143
1: -17.8925095, -15.6412506, -17.9268627, -15.6723633, -1.3899145, 1.3780808
2: -6.5690851, -4.6015534, -6.5183969, -4.5250416, -1.4514756, 1.3429651
3: -13.9449673, -12.1600914, -13.9528704, -12.1378860, -1.2461982, 1.2326212
4: -5.5172033, -3.7015522, -5.5817490, -3.7246513, -1.7028513, 1.7506928
5: -7.0068040, -5.5910363, -7.0356169, -5.6153336, -0.9634364, 1.0076108
6: 8.3495007, 10.0807190, 8.3106194, 10.0272884, -1.2144737, 1.3407960
7: -14.0296831, -12.2535391, -14.0085449, -12.1836500, -1.0712950, 0.9839189
8: -6.0298171, -4.6201839, -6.0787020, -4.6586089, -0.7805538, 0.8727269
9: -10.7607145, -8.5254669, -10.8038044, -8.5102825, -1.7153664, 1.7657342

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A2_B1_A1_B2_A1_B1

### Relational analysis result of IS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7691395, upper bound: 0.7548528
time: 4.32 seconds

## Relational analysis of IS_A2_B1_A1_B2_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7697928, upper bound: 0.7565066
time: 8.16 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 18.30 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7743129, upper bound: 0.7657671
IS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7749489, upper bound: 0.7667032
IS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7760745, upper bound: 0.7758004
IS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7766439, upper bound: 0.7773962
IS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7752398, upper bound: 0.7630020
IS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7758890, upper bound: 0.7642236
IS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7762672, upper bound: 0.7714616
IS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7768217, upper bound: 0.7730680
IS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7794602, upper bound: 0.7806615
IS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7794602, upper bound: 0.7807185
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7808013, upper bound: 0.7916357
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7808013, upper bound: 0.7923075
IS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7803615, upper bound: 0.7782257
IS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7803615, upper bound: 0.7783475
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7809150, upper bound: 0.7873106
IS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7809150, upper bound: 0.7880372
IS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7667408, upper bound: 0.7589413
IS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7676548, upper bound: 0.7594594
IS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7693355, upper bound: 0.7723244
IS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7707454, upper bound: 0.7740954
IS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7677953, upper bound: 0.7572398
IS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7686620, upper bound: 0.7577338
IS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7697537, upper bound: 0.7681304
IS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7711624, upper bound: 0.7699566
IS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7715192, upper bound: 0.7730563
IS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7715192, upper bound: 0.7731167
IS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7743341, upper bound: 0.7879716
IS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7743341, upper bound: 0.7887252
IS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7725058, upper bound: 0.7713727
IS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7725058, upper bound: 0.7714268
IS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7747492, upper bound: 0.7838551
IS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7747492, upper bound: 0.7846709
IS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7680953, upper bound: 0.7570455
IS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7687321, upper bound: 0.7587523
IS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7700981, upper bound: 0.7688013
IS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7717527, upper bound: 0.7714507
IS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7691395, upper bound: 0.7548528
IS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 18.30
Output dim: 6, lower bound: -0.7697928, upper bound: 0.7565066
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7838529, upper bound: 0.7747495
IS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7803044, upper bound: 0.7731032
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7842377, upper bound: 0.7866189
IS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7813634, upper bound: 0.7709109
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7846686, upper bound: 0.7824484
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7730564, upper bound: 0.7687034
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7713726, upper bound: 0.7697234
IS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7879713, upper bound: 0.7719247
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7838530, upper bound: 0.7723402
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7803044, upper bound: 0.7684073
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7842377, upper bound: 0.7827127
IS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7813634, upper bound: 0.7664520
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.30
Output dim: 6, lower bound: -0.7846686, upper bound: 0.7785803
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.3650121688842773
rel_dist={6: [-0.846481615748317, 0.8464836721348625]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2130
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 2130

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7019103, upper bound: 0.7061180
time: 5.99 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7061178, upper bound: 0.7061181
time: 4.54 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.82 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.82
Output dim: 6, lower bound: -0.7019103, upper bound: 0.7061180
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.82
Output dim: 6, lower bound: -0.7061178, upper bound: 0.7061181

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -1.6506550, 0.1124235, -1.6551847, 0.1125117, -1.4416652, 1.4483171
1: -17.9605484, -15.6270342, -17.9635086, -15.6268339, -1.4455905, 1.4485257
2: -6.5349741, -4.4834495, -6.5349874, -4.4655275, -1.3666172, 1.3468962
3: -13.9778080, -12.1164856, -13.9783020, -12.1088562, -1.2286515, 1.2026696
4: -5.6252713, -3.7163892, -5.6351042, -3.7163746, -1.7583551, 1.7721019
5: -7.0521522, -5.5889454, -7.0535679, -5.5888672, -0.9814594, 0.9788480
6: 8.2744751, 10.0458050, 8.2598763, 10.0458174, -1.2687058, 1.2780724
7: -14.0097361, -12.1294889, -14.0097408, -12.1193342, -1.0567510, 1.0516400
8: -6.1093316, -4.6512423, -6.1145868, -4.6512351, -0.8169184, 0.8190001
9: -10.8447342, -8.5088797, -10.8449192, -8.5054426, -1.7566543, 1.7554078

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7001077, upper bound: 0.7001073
time: 4.71 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7001077, upper bound: 0.7061176
time: 4.65 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -1.6475005, 0.1262470, -1.6506941, 0.1123602, -1.4397049, 1.4670010
1: -17.9519997, -15.6137323, -17.9592533, -15.6271057, -1.4748497, 1.4642019
2: -6.5853863, -4.5002308, -6.5349827, -4.4805493, -1.4337053, 1.3532748
3: -14.0151386, -12.1148701, -13.9775867, -12.1103973, -1.2737713, 1.2339711
4: -5.6131539, -3.6883640, -5.6260395, -3.7163842, -1.7577810, 1.8062983
5: -7.0425291, -5.5800605, -7.0492816, -5.5889940, -1.0099454, 0.9760969
6: 8.3123417, 10.1139135, 8.2845974, 10.0457954, -1.2357249, 1.3303547
7: -14.0357504, -12.1219559, -14.0097351, -12.1239500, -1.0750055, 1.0435209
8: -6.1018839, -4.6322112, -6.1074400, -4.6512494, -0.8047380, 0.8111567
9: -10.8561497, -8.5267410, -10.8446274, -8.5145483, -1.7475204, 1.7524295

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7061180, upper bound: 0.7001102
time: 4.05 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7061180, upper bound: 0.7061204
time: 4.06 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 13.86 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 13.86
Output dim: 6, lower bound: -0.7001077, upper bound: 0.7001073
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 13.86
Output dim: 6, lower bound: -0.7001077, upper bound: 0.7061176
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 13.86
Output dim: 6, lower bound: -0.7061180, upper bound: 0.7001102
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 13.86
Output dim: 6, lower bound: -0.7061180, upper bound: 0.7061204

## BFS IS instance: IS_A1_B1

### Backsubstitution after applying IS history:
0: -1.6506550, 0.1124235, -1.6506550, 0.1124235, -1.4406972, 1.4406972
1: -17.9605484, -15.6270342, -17.9605484, -15.6270342, -1.4388218, 1.4388220
2: -6.5349741, -4.4834495, -6.5349741, -4.4834495, -1.3468843, 1.3468843
3: -13.9778080, -12.1164856, -13.9778080, -12.1164856, -1.1927900, 1.1927900
4: -5.6252713, -3.7163892, -5.6252713, -3.7163892, -1.7582045, 1.7582045
5: -7.0521522, -5.5889454, -7.0521522, -5.5889454, -0.9734018, 0.9734019
6: 8.2744751, 10.0458050, 8.2744751, 10.0458050, -1.2686901, 1.2686896
7: -14.0097361, -12.1294889, -14.0097361, -12.1294889, -1.0516379, 1.0516374
8: -6.1093316, -4.6512423, -6.1093316, -4.6512423, -0.8169129, 0.8169129
9: -10.8447342, -8.5088797, -10.8447342, -8.5088797, -1.7553124, 1.7553129

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6954817, upper bound: 0.6890772
time: 5.73 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6963859, upper bound: 0.6941957
time: 3.78 seconds

## BFS IS instance: IS_A1_B2

### Backsubstitution after applying IS history:
0: -1.6506550, 0.1124235, -1.6475005, 0.1262470, -1.4578872, 1.4408751
1: -17.9605484, -15.6270342, -17.9519997, -15.6137323, -1.4560881, 1.4353518
2: -6.5349741, -4.4834495, -6.5853863, -4.5002308, -1.3624964, 1.4172363
3: -13.9778080, -12.1164856, -14.0151386, -12.1148701, -1.1995010, 1.2337594
4: -5.6252713, -3.7163892, -5.6131539, -3.6883640, -1.7962494, 1.7604446
5: -7.0521522, -5.5889454, -7.0425291, -5.5800605, -0.9800127, 0.9657294
6: 8.2744751, 10.0458050, 8.3123417, 10.1139135, -1.3509965, 1.2273450
7: -14.0097361, -12.1294889, -14.0357504, -12.1219559, -1.0432343, 1.0767498
8: -6.1093316, -4.6512423, -6.1018839, -4.6322112, -0.8195608, 0.7990724
9: -10.8447342, -8.5088797, -10.8561497, -8.5267410, -1.7285366, 1.7578912

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6954817, upper bound: 0.6950700
time: 6.23 seconds

## Relational analysis of IS_A1_B2_A2

### Relational analysis result of IS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6963859, upper bound: 0.7001029
time: 3.97 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -1.6475005, 0.1262470, -1.6506550, 0.1124235, -1.4408755, 1.4578872
1: -17.9519997, -15.6137323, -17.9605484, -15.6270342, -1.4353518, 1.4560881
2: -6.5853863, -4.5002308, -6.5349741, -4.4834495, -1.4172363, 1.3624964
3: -14.0151386, -12.1148701, -13.9778080, -12.1164856, -1.2337594, 1.1995010
4: -5.6131539, -3.6883640, -5.6252713, -3.7163892, -1.7604446, 1.7962494
5: -7.0425291, -5.5800605, -7.0521522, -5.5889454, -0.9657292, 0.9800125
6: 8.3123417, 10.1139135, 8.2744751, 10.0458050, -1.2273450, 1.3509965
7: -14.0357504, -12.1219559, -14.0097361, -12.1294889, -1.0767496, 1.0432341
8: -6.1018839, -4.6322112, -6.1093316, -4.6512423, -0.7990723, 0.8195608
9: -10.8561497, -8.5267410, -10.8447342, -8.5088797, -1.7578912, 1.7285366

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A2_B1_A1

### Relational analysis result of IS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6990501, upper bound: 0.6890766
time: 4.38 seconds

## Relational analysis of IS_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7001016, upper bound: 0.6941944
time: 3.97 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -1.6475005, 0.1262470, -1.6475005, 0.1262470, -1.4634953, 1.4634953
1: -17.9519997, -15.6137323, -17.9519997, -15.6137323, -1.4773955, 1.4773955
2: -6.5853863, -4.5002308, -6.5853863, -4.5002308, -1.3591514, 1.3591514
3: -14.0151386, -12.1148701, -14.0151386, -12.1148701, -1.2124891, 1.2124891
4: -5.6131539, -3.6883640, -5.6131539, -3.6883640, -1.7572927, 1.7572932
5: -7.0425291, -5.5800605, -7.0425291, -5.5800605, -1.0103412, 1.0103412
6: 8.3123417, 10.1139135, 8.3123417, 10.1139135, -1.2536163, 1.2536161
7: -14.0357504, -12.1219559, -14.0357504, -12.1219559, -1.0672824, 1.0672824
8: -6.1018839, -4.6322112, -6.1018839, -4.6322112, -0.8034950, 0.8034950
9: -10.8561497, -8.5267410, -10.8561497, -8.5267410, -1.7519908, 1.7519908

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6990503, upper bound: 0.6890764
time: 7.43 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7001018, upper bound: 0.6941944
time: 4.21 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 17.38 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.38
Output dim: 6, lower bound: -0.6954817, upper bound: 0.6890772
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.38
Output dim: 6, lower bound: -0.6963859, upper bound: 0.6941957
IS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.38
Output dim: 6, lower bound: -0.6954817, upper bound: 0.6950700
IS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.38
Output dim: 6, lower bound: -0.6963859, upper bound: 0.7001029
IS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.38
Output dim: 6, lower bound: -0.6990501, upper bound: 0.6890766
IS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.38
Output dim: 6, lower bound: -0.7001016, upper bound: 0.6941944
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.38
Output dim: 6, lower bound: -0.6990503, upper bound: 0.6890764
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.38
Output dim: 6, lower bound: -0.7001018, upper bound: 0.6941944

## BFS IS instance: IS_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6440289, 0.1170566, -1.6475470, 0.1124224, -1.4289289, 1.4356947
1: -17.9686394, -15.6648798, -17.9605503, -15.6455193, -1.4218082, 1.3938756
2: -6.5292134, -4.4837713, -6.5322599, -4.4836016, -1.3403606, 1.3433518
3: -13.9683418, -12.1419897, -13.9777946, -12.1284962, -1.1772203, 1.1740928
4: -5.5857277, -3.7300534, -5.6066494, -3.7179618, -1.7197390, 1.7301784
5: -7.0527544, -5.6105976, -7.0509210, -5.6002588, -0.9429429, 0.9389172
6: 8.2773838, 10.0093632, 8.2744913, 10.0280552, -1.2364268, 1.2293048
7: -14.0086727, -12.1354914, -14.0092344, -12.1322794, -1.0477300, 1.0451083
8: -6.0861721, -4.6511135, -6.0983963, -4.6512451, -0.7916219, 0.8024997
9: -10.8196745, -8.5050039, -10.8329639, -8.5088806, -1.7246122, 1.7417107

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6961075, upper bound: 0.6951557
time: 4.44 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6963398, upper bound: 0.6917284
time: 4.04 seconds

## BFS IS instance: IS_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6485668, 0.1124225, -1.4312167, 1.4396105
1: -17.9605503, -15.6407909, -17.9605503, -15.6321545, -1.4347329, 1.4121208
2: -6.5333314, -4.4837227, -6.5343671, -4.4835496, -1.3452630, 1.3465528
3: -13.9777908, -12.1190071, -13.9778004, -12.1174812, -1.1915288, 1.1751709
4: -5.6204309, -3.7172384, -5.6234894, -3.7167022, -1.7231388, 1.7533736
5: -7.0514894, -5.5967188, -7.0519099, -5.5918832, -0.9700527, 0.9326035
6: 8.2744904, 10.0354328, 8.2744808, 10.0419827, -1.2674432, 1.2241764
7: -14.0094528, -12.1319809, -14.0096331, -12.1304121, -1.0491042, 1.0483863
8: -6.1021070, -4.6512437, -6.1066675, -4.6512427, -0.7963414, 0.8144101
9: -10.8361607, -8.5088787, -10.8409233, -8.5088787, -1.7411795, 1.7527485

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6975020, upper bound: 0.7011978
time: 4.37 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6977468, upper bound: 0.6977488
time: 4.29 seconds

## BFS IS instance: IS_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6440289, 0.1170566, -1.6444678, 0.1262453, -1.4461184, 1.4358425
1: -17.9686394, -15.6648798, -17.9520016, -15.6324778, -1.4394970, 1.3904047
2: -6.5292134, -4.4837713, -6.5826735, -4.5003815, -1.3559737, 1.4137211
3: -13.9683418, -12.1419897, -14.0151243, -12.1268787, -1.1839314, 1.2150559
4: -5.5857277, -3.7300534, -5.5939283, -3.6899219, -1.7578144, 1.7323947
5: -7.0527544, -5.6105976, -7.0412989, -5.5913754, -0.9500494, 0.9312693
6: 8.2773838, 10.0093632, 8.3123569, 10.0967646, -1.3193526, 1.1879601
7: -14.0086727, -12.1354914, -14.0352478, -12.1247196, -1.0392413, 1.0702405
8: -6.0861721, -4.6511135, -6.0910091, -4.6322117, -0.7942705, 0.7846311
9: -10.8196745, -8.5050039, -10.8440924, -8.5267410, -1.6978364, 1.7437768

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_A1_B1

### Relational analysis result of IS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6869824, upper bound: 0.6904700
time: 5.81 seconds

## Relational analysis of IS_A1_B2_A1_B2

### Relational analysis result of IS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6874145, upper bound: 0.6869500
time: 6.33 seconds

## BFS IS instance: IS_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6454366, 0.1262460, -1.4484038, 1.4397840
1: -17.9605503, -15.6407909, -17.9519997, -15.6188765, -1.4516954, 1.4100504
2: -6.5333314, -4.4837227, -6.5847778, -4.5003300, -1.3608756, 1.4168377
3: -13.9777908, -12.1190071, -14.0151329, -12.1158628, -1.1982398, 1.2157454
4: -5.6204309, -3.7172384, -5.6116214, -3.6886718, -1.7611837, 1.7556133
5: -7.0514894, -5.5967188, -7.0422869, -5.5829749, -0.9765055, 0.9249369
6: 8.2744904, 10.0354328, 8.3123455, 10.1098127, -1.3494725, 1.1845274
7: -14.0094528, -12.1319809, -14.0356445, -12.1228819, -1.0408878, 1.0734904
8: -6.1021070, -4.6512437, -6.0992312, -4.6322107, -0.7988987, 0.7966013
9: -10.8361607, -8.5088787, -10.8522806, -8.5267420, -1.7144027, 1.7552285

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6878688, upper bound: 0.6954485
time: 6.26 seconds

## Relational analysis of IS_A1_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6882989, upper bound: 0.6919693
time: 6.13 seconds

## BFS IS instance: IS_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6475470, 0.1124224, -1.4290123, 1.4528823
1: -17.9600925, -15.6518440, -17.9605503, -15.6455193, -1.4197369, 1.4120073
2: -6.5795999, -4.5005484, -6.5322599, -4.4836016, -1.4107108, 1.3589664
3: -14.0056801, -12.1403723, -13.9777946, -12.1284962, -1.2177796, 1.1808028
4: -5.5723267, -3.7033584, -5.6066494, -3.7179618, -1.7219563, 1.7682824
5: -7.0431242, -5.6016593, -7.0509210, -5.6002588, -0.9353228, 0.9464414
6: 8.3133726, 10.0787506, 8.2744913, 10.0280552, -1.1967859, 1.3129230
7: -14.0346870, -12.1278963, -14.0092344, -12.1322794, -1.0728846, 1.0367134
8: -6.0788412, -4.6320820, -6.0983963, -4.6512451, -0.7738087, 0.8050568
9: -10.8304892, -8.5228643, -10.8329639, -8.5088806, -1.7261181, 1.7149339

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_B1_A1_B1

### Relational analysis result of IS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6904991, upper bound: 0.6858901
time: 5.65 seconds

## Relational analysis of IS_A2_B1_A1_B2

### Relational analysis result of IS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6909303, upper bound: 0.6824400
time: 5.00 seconds

## BFS IS instance: IS_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6485668, 0.1124225, -1.4313626, 1.4568000
1: -17.9519997, -15.6275787, -17.9605503, -15.6321545, -1.4312644, 1.4303164
2: -6.5837455, -4.5005035, -6.5343671, -4.4835496, -1.4155278, 1.3621645
3: -14.0151215, -12.1173935, -13.9778004, -12.1174812, -1.2324891, 1.1818814
4: -5.6089945, -3.6892047, -5.6234894, -3.7167022, -1.7253551, 1.7914329
5: -7.0418663, -5.5878100, -7.0519099, -5.5918832, -0.9623914, 0.9401283
6: 8.3123579, 10.1027765, 8.2744808, 10.0419827, -1.2260985, 1.3077216
7: -14.0354652, -12.1244431, -14.0096331, -12.1304121, -1.0742555, 1.0401585
8: -6.0946827, -4.6322107, -6.1066675, -4.6512427, -0.7784805, 0.8170567
9: -10.8474274, -8.5267429, -10.8409233, -8.5088787, -1.7434864, 1.7259717

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6915695, upper bound: 0.6917805
time: 6.26 seconds

## Relational analysis of IS_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6919690, upper bound: 0.6883011
time: 4.54 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6444678, 0.1262453, -1.4516320, 1.4584589
1: -17.9600925, -15.6518440, -17.9520016, -15.6324778, -1.4598494, 1.4326541
2: -6.5795999, -4.5005484, -6.5826735, -4.5003815, -1.3526368, 1.3556442
3: -14.0056801, -12.1403723, -14.0151243, -12.1268787, -1.1981516, 1.1946039
4: -5.5723267, -3.7033584, -5.5939283, -3.6899219, -1.7188025, 1.7292442
5: -7.0431242, -5.6016593, -7.0412989, -5.5913754, -0.9784038, 0.9757872
6: 8.3133726, 10.0787506, 8.3123569, 10.0967646, -1.2209539, 1.2142460
7: -14.0346870, -12.1278963, -14.0352478, -12.1247196, -1.0633311, 1.0607817
8: -6.0788412, -4.6320820, -6.0910091, -4.6322117, -0.7781655, 0.7890210
9: -10.8304892, -8.5228643, -10.8440924, -8.5267410, -1.7212963, 1.7383885

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6904991, upper bound: 0.6844730
time: 4.42 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6909303, upper bound: 0.6809735
time: 4.50 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6454366, 0.1262460, -1.4539790, 1.4624028
1: -17.9519997, -15.6275787, -17.9519997, -15.6188765, -1.4733949, 1.4500139
2: -6.5837455, -4.5005035, -6.5847778, -4.5003300, -1.3574462, 1.3587542
3: -14.0151215, -12.1173935, -14.0151329, -12.1158628, -1.2110658, 1.1965156
4: -5.6089945, -3.6892047, -5.6116214, -3.6886718, -1.7222071, 1.7524567
5: -7.0418663, -5.5878100, -7.0422869, -5.5829749, -1.0069947, 0.9677522
6: 8.3123579, 10.1027765, 8.3123455, 10.1098127, -1.2523522, 1.2088418
7: -14.0354652, -12.1244431, -14.0356445, -12.1228819, -1.0649729, 1.0641987
8: -6.0946827, -4.6322107, -6.0992312, -4.6322107, -0.7828834, 0.8010174
9: -10.8474274, -8.5267429, -10.8522806, -8.5267420, -1.7378616, 1.7494268

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6915695, upper bound: 0.6895899
time: 7.65 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6919690, upper bound: 0.6860807
time: 6.23 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.67 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6961075, upper bound: 0.6951557
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6963398, upper bound: 0.6917284
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6975020, upper bound: 0.7011978
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6977468, upper bound: 0.6977488
IS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6869824, upper bound: 0.6904700
IS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6874145, upper bound: 0.6869500
IS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6878688, upper bound: 0.6954485
IS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6882989, upper bound: 0.6919693
IS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6904991, upper bound: 0.6858901
IS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6909303, upper bound: 0.6824400
IS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6915695, upper bound: 0.6917805
IS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6919690, upper bound: 0.6883011
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6904991, upper bound: 0.6844730
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6909303, upper bound: 0.6809735
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6915695, upper bound: 0.6895899
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 6, lower bound: -0.6919690, upper bound: 0.6860807

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6432234, 0.1128181, -1.6499729, 0.1031393, -1.4124923, 1.4303989
1: -17.9685860, -15.6807880, -17.9641285, -15.6809921, -1.3844929, 1.3765159
2: -6.5276890, -4.4847841, -6.5288696, -4.4791889, -1.3389568, 1.3354735
3: -13.9677668, -12.1460953, -13.9849825, -12.1376066, -1.1650934, 1.1738243
4: -5.5592690, -3.7308421, -5.5499783, -3.7334421, -1.6662836, 1.6770267
5: -7.0517507, -5.6146441, -7.0614443, -5.6092196, -0.9296093, 0.9412113
6: 8.2858191, 10.0093079, 8.2927990, 10.0427694, -1.2234626, 1.2002258
7: -14.0081520, -12.1358147, -14.0081310, -12.1330109, -1.0460813, 1.0435531
8: -6.0840697, -4.6600170, -6.0929451, -4.6709428, -0.7664322, 0.7840004
9: -10.8109722, -8.5055037, -10.8139410, -8.5148392, -1.7111669, 1.7215681

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6833010, upper bound: 0.6757284
time: 5.91 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6863915, upper bound: 0.6852173
time: 4.22 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.6436043, 0.1136424, -1.6464969, 0.1040422, -1.4236083, 1.4267397
1: -17.9685364, -15.6736002, -17.9602871, -15.6657143, -1.4006243, 1.3815415
2: -6.5251656, -4.4843855, -6.5222340, -4.4850283, -1.3367338, 1.3361883
3: -13.9680023, -12.1467810, -13.9770126, -12.1403608, -1.1709938, 1.1698799
4: -5.5823302, -3.7303810, -5.5992317, -3.7188237, -1.7005844, 1.6713428
5: -7.0522585, -5.6176691, -7.0497088, -5.6177616, -0.9385715, 0.9330738
6: 8.2876568, 10.0093288, 8.2976646, 10.0279760, -1.2282810, 1.2044554
7: -14.0083742, -12.1356468, -14.0084963, -12.1326571, -1.0463903, 1.0446079
8: -6.0852809, -4.6540737, -6.0962262, -4.6585512, -0.7745532, 0.7946186
9: -10.8184319, -8.5052147, -10.8299856, -8.5093861, -1.7115259, 1.7406774

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6837531, upper bound: 0.6734059
time: 4.25 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6865726, upper bound: 0.6819523
time: 3.73 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.6442518, 0.1081837, -1.6510073, 0.1031384, -1.4147663, 1.4343534
1: -17.9604950, -15.6568413, -17.9641266, -15.6677275, -1.3974619, 1.3947589
2: -6.5318069, -4.4847345, -6.5309772, -4.4791374, -1.3438587, 1.3386545
3: -13.9772177, -12.1231136, -13.9849882, -12.1265888, -1.1794000, 1.1749034
4: -5.5939741, -3.7180874, -5.5668182, -3.7322001, -1.6697025, 1.7002487
5: -7.0504398, -5.6007576, -7.0624757, -5.6008353, -0.9566829, 0.9349638
6: 8.2829065, 10.0353765, 8.2927904, 10.0567179, -1.2545013, 1.1951241
7: -14.0089340, -12.1323023, -14.0085268, -12.1311426, -1.0474443, 1.0468247
8: -6.1000195, -4.6601458, -6.1012149, -4.6709423, -0.7711650, 0.7960429
9: -10.8274574, -8.5093746, -10.8219032, -8.5148373, -1.7277632, 1.7326069

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6839958, upper bound: 0.6807937
time: 4.70 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6876894, upper bound: 0.6911633
time: 4.57 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.6446304, 0.1090070, -1.6475129, 0.1040426, -1.4259586, 1.4306774
1: -17.9604416, -15.6494045, -17.9602871, -15.6522579, -1.4135170, 1.3997712
2: -6.5292845, -4.4843364, -6.5243435, -4.4849749, -1.3416710, 1.3393888
3: -13.9774551, -12.1237984, -13.9770174, -12.1293430, -1.1853013, 1.1709619
4: -5.6170349, -3.7175918, -5.6160712, -3.7175689, -1.7039661, 1.6946678
5: -7.0509739, -5.6037850, -7.0506964, -5.6093798, -0.9656427, 0.9267979
6: 8.2847528, 10.0353994, 8.2976522, 10.0419025, -1.2592902, 1.1994247
7: -14.0091524, -12.1321373, -14.0088930, -12.1307917, -1.0477571, 1.0478818
8: -6.1012220, -4.6542034, -6.1045156, -4.6585479, -0.7794576, 0.8065456
9: -10.8349180, -8.5090876, -10.8379469, -8.5093880, -1.7281013, 1.7516551

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6845709, upper bound: 0.6785324
time: 4.55 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6879671, upper bound: 0.6879667
time: 3.71 seconds

## BFS IS instance: IS_A1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6432234, 0.1128181, -1.6466364, 0.1169794, -1.4296656, 1.4303350
1: -17.9685860, -15.6807880, -17.9555759, -15.6681948, -1.4018278, 1.3730085
2: -6.5276890, -4.4847841, -6.5792837, -4.4959326, -1.3545437, 1.4058418
3: -13.9677668, -12.1460953, -14.0223799, -12.1359901, -1.1717873, 1.2146196
4: -5.5592690, -3.7308421, -5.5373545, -3.7054191, -1.7043867, 1.6793270
5: -7.0517507, -5.6146441, -7.0518346, -5.6002846, -0.9369638, 0.9336053
6: 8.2858191, 10.0093079, 8.3304386, 10.1114826, -1.3063841, 1.1591420
7: -14.0081520, -12.1358147, -14.0341473, -12.1254110, -1.0376401, 1.0687001
8: -6.0840697, -4.6600170, -6.0854859, -4.6519098, -0.7690921, 0.7659408
9: -10.8109722, -8.5055037, -10.8251829, -8.5326490, -1.6843958, 1.7236977

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_A1_B1_A1

### Relational analysis result of IS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6734597, upper bound: 0.6676928
time: 4.54 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2

### Relational analysis result of IS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6778428, upper bound: 0.6803934
time: 4.13 seconds

## BFS IS instance: IS_A1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6436043, 0.1136424, -1.6434022, 0.1178820, -1.4407973, 1.4266729
1: -17.9685364, -15.6736002, -17.9517345, -15.6517048, -1.4179015, 1.3780360
2: -6.5251656, -4.4843855, -6.5726480, -4.5017834, -1.3523493, 1.4065580
3: -13.9680023, -12.1467810, -14.0143185, -12.1387472, -1.1776981, 1.2107987
4: -5.5823302, -3.7303810, -5.5864801, -3.6907902, -1.7386723, 1.6736116
5: -7.0522585, -5.6176691, -7.0400844, -5.6083727, -0.9458995, 0.9254684
6: 8.2876568, 10.0093288, 8.3357964, 10.0966883, -1.3112049, 1.1631050
7: -14.0083742, -12.1356468, -14.0345125, -12.1250725, -1.0379257, 1.0697520
8: -6.0852809, -4.6540737, -6.0889006, -4.6395168, -0.7772126, 0.7765009
9: -10.8184319, -8.5052147, -10.8411627, -8.5272207, -1.6847525, 1.7428007

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_A1_B2_A1

### Relational analysis result of IS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6741639, upper bound: 0.6661273
time: 3.99 seconds

## Relational analysis of IS_A1_B2_A1_B2_A2

### Relational analysis result of IS_A1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6781471, upper bound: 0.6770879
time: 4.47 seconds

## BFS IS instance: IS_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6442518, 0.1081837, -1.6476203, 0.1169798, -1.4319358, 1.4343152
1: -17.9604950, -15.6568413, -17.9555779, -15.6547756, -1.4139857, 1.3926530
2: -6.5318069, -4.4847345, -6.5813909, -4.4958820, -1.3594456, 1.4089398
3: -13.9772177, -12.1231136, -14.0223885, -12.1249704, -1.1860943, 1.2152982
4: -5.5939741, -3.7180874, -5.5550489, -3.7041898, -1.7077713, 1.7025723
5: -7.0504398, -5.6007576, -7.0528660, -5.5918775, -0.9633832, 0.9273397
6: 8.2829065, 10.0353765, 8.3304272, 10.1245461, -1.3365245, 1.1557367
7: -14.0089340, -12.1323023, -14.0345459, -12.1235714, -1.0392997, 1.0719430
8: -6.1000195, -4.6601458, -6.0937371, -4.6519098, -0.7737333, 0.7780597
9: -10.8274574, -8.5093746, -10.8333712, -8.5326481, -1.7009907, 1.7351508

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_A2_B1_A1

### Relational analysis result of IS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6741140, upper bound: 0.6723569
time: 4.40 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2

### Relational analysis result of IS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6787417, upper bound: 0.6853621
time: 4.33 seconds

## BFS IS instance: IS_A1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6446304, 0.1090070, -1.6443675, 0.1178832, -1.4431453, 1.4306517
1: -17.9604416, -15.6494045, -17.9517345, -15.6380920, -1.4300315, 1.3976688
2: -6.5292845, -4.4843364, -6.5747538, -4.5017309, -1.3572860, 1.4096737
3: -13.9774551, -12.1237984, -14.0143270, -12.1277294, -1.1920061, 1.2114835
4: -5.6170349, -3.7175918, -5.6041737, -3.6895466, -1.7420206, 1.6969604
5: -7.0509739, -5.6037850, -7.0410724, -5.5999684, -0.9723167, 0.9191746
6: 8.2847528, 10.0353994, 8.3357878, 10.1097345, -1.3413172, 1.1597710
7: -14.0091524, -12.1321373, -14.0349083, -12.1232338, -1.0395765, 1.0729983
8: -6.1012220, -4.6542034, -6.0971365, -4.6395159, -0.7820250, 0.7884743
9: -10.8349180, -8.5090876, -10.8493528, -8.5272198, -1.7013283, 1.7541900

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_A2_B2_A1

### Relational analysis result of IS_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6749432, upper bound: 0.6708266
time: 3.88 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2

### Relational analysis result of IS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6791011, upper bound: 0.6821525
time: 3.97 seconds

## BFS IS instance: IS_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6402003, 0.1266484, -1.6499729, 0.1031393, -1.4125533, 1.4475765
1: -17.9600353, -15.6679697, -17.9641285, -15.6809921, -1.3824158, 1.3944693
2: -6.5780759, -4.5015416, -6.5288696, -4.4791889, -1.4093075, 1.3511038
3: -14.0050936, -12.1444807, -13.9849825, -12.1376066, -1.2056279, 1.1805291
4: -5.5459003, -3.7041667, -5.5499783, -3.7334421, -1.6685390, 1.7151423
5: -7.0421190, -5.6056967, -7.0614443, -5.6092196, -0.9220314, 0.9488232
6: 8.3216934, 10.0786972, 8.2927990, 10.0427694, -1.1838999, 1.2838421
7: -14.0341682, -12.1281948, -14.0081310, -12.1330109, -1.0712402, 1.0351820
8: -6.0767961, -4.6409826, -6.0929451, -4.6709428, -0.7487282, 0.7865589
9: -10.8218288, -8.5233374, -10.8139410, -8.5148392, -1.7126946, 1.6947927

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6771497, upper bound: 0.6651164
time: 4.80 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6807602, upper bound: 0.6764525
time: 4.63 seconds

## BFS IS instance: IS_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.6405871, 0.1274721, -1.6464969, 0.1040422, -1.4235258, 1.4439173
1: -17.9599857, -15.6599989, -17.9602871, -15.6657143, -1.3985405, 1.4002647
2: -6.5755539, -4.5011525, -6.5222340, -4.4850283, -1.4070854, 1.3518038
3: -14.0053329, -12.1451664, -13.9770126, -12.1403608, -1.2114201, 1.1765885
4: -5.5689421, -3.7036924, -5.5992317, -3.7188237, -1.7028046, 1.7094555
5: -7.0426278, -5.6086063, -7.0497088, -5.6177616, -0.9309177, 0.9405112
6: 8.3239346, 10.0787172, 8.2976646, 10.0279760, -1.1888146, 1.2880750
7: -14.0343885, -12.1280365, -14.0084963, -12.1326571, -1.0715501, 1.0362225
8: -6.0779772, -4.6350412, -6.0962262, -4.6585512, -0.7565087, 0.7971789
9: -10.8292656, -8.5230637, -10.8299856, -8.5093861, -1.7130518, 1.7139020

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6780762, upper bound: 0.6630160
time: 7.28 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6810679, upper bound: 0.6731594
time: 4.48 seconds

## BFS IS instance: IS_A2_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.6411318, 0.1220149, -1.6510073, 0.1031384, -1.4149070, 1.4515343
1: -17.9519424, -15.6439171, -17.9641266, -15.6677275, -1.3939853, 1.4125967
2: -6.5822186, -4.5014949, -6.5309772, -4.4791374, -1.4141235, 1.3542829
3: -14.0145350, -12.1214981, -13.9849882, -12.1265888, -1.2203417, 1.1816096
4: -5.5825682, -3.6900628, -5.5668182, -3.7322001, -1.6719599, 1.7383184
5: -7.0408154, -5.5918417, -7.0624757, -5.6008353, -0.9490628, 0.9425764
6: 8.3206587, 10.1027184, 8.2927904, 10.0567179, -1.2132354, 1.2786670
7: -14.0349464, -12.1247444, -14.0085268, -12.1311426, -1.0725985, 1.0386319
8: -6.0926561, -4.6411138, -6.1012149, -4.6709423, -0.7534143, 0.7986914
9: -10.8387680, -8.5272064, -10.8219032, -8.5148373, -1.7300930, 1.7058325

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B1_A2_B1_A1

### Relational analysis result of IS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6779142, upper bound: 0.6707123
time: 4.77 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2

### Relational analysis result of IS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6817549, upper bound: 0.6823214
time: 4.57 seconds

## BFS IS instance: IS_A2_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.6415184, 0.1228378, -1.6475129, 0.1040426, -1.4259477, 1.4478574
1: -17.9518929, -15.6355352, -17.9602871, -15.6522579, -1.4100342, 1.4185586
2: -6.5796952, -4.5011053, -6.5243435, -4.4849749, -1.4119358, 1.3550019
3: -14.0147734, -12.1221828, -13.9770174, -12.1293430, -1.2261329, 1.1776695
4: -5.6056080, -3.6895614, -5.6160712, -3.7175689, -1.7061872, 1.7327356
5: -7.0413523, -5.5947528, -7.0506964, -5.6093798, -0.9579473, 0.9342360
6: 8.3229103, 10.1027412, 8.2976522, 10.0419025, -1.2181196, 1.2829680
7: -14.0351696, -12.1245842, -14.0088930, -12.1307917, -1.0729122, 1.0396674
8: -6.0938253, -4.6351709, -6.1045156, -4.6585479, -0.7613769, 0.8091958
9: -10.8462067, -8.5269384, -10.8379469, -8.5093880, -1.7304273, 1.7248797

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B1_A2_B2_A1

### Relational analysis result of IS_A2_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6788682, upper bound: 0.6686686
time: 5.41 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2

### Relational analysis result of IS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6821522, upper bound: 0.6791016
time: 7.62 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6402003, 0.1266484, -1.6466364, 0.1169794, -1.4351544, 1.4529433
1: -17.9600353, -15.6679697, -17.9555759, -15.6681948, -1.4224582, 1.4152772
2: -6.5780759, -4.5015416, -6.5792837, -4.4959326, -1.3512444, 1.3477902
3: -14.0050936, -12.1444807, -14.0223799, -12.1359901, -1.1860843, 1.1942639
4: -5.5459003, -3.7041667, -5.5373545, -3.7054191, -1.6653662, 1.6761231
5: -7.0421190, -5.6056967, -7.0518346, -5.6002846, -0.9652569, 0.9781089
6: 8.3216934, 10.0786972, 8.3304386, 10.1114826, -1.2077417, 1.1848392
7: -14.0341682, -12.1281948, -14.0341473, -12.1254110, -1.0617340, 1.0592642
8: -6.0767961, -4.6409826, -6.0854859, -4.6519098, -0.7529786, 0.7701116
9: -10.8218288, -8.5233374, -10.8251829, -8.5326490, -1.7078404, 1.7182193

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6771497, upper bound: 0.6630655
time: 4.83 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6807605, upper bound: 0.6749819
time: 4.75 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6405871, 0.1274721, -1.6434022, 0.1178820, -1.4461432, 1.4492807
1: -17.9599857, -15.6599989, -17.9517345, -15.6517048, -1.4384079, 1.4209893
2: -6.5755539, -4.5011525, -6.5726480, -4.5017834, -1.3490443, 1.3484926
3: -14.0053329, -12.1451664, -14.0143185, -12.1387472, -1.1918516, 1.1904206
4: -5.5689421, -3.7036924, -5.5864801, -3.6907902, -1.6996355, 1.6704330
5: -7.0426278, -5.6086063, -7.0400844, -5.6083727, -0.9741695, 0.9698482
6: 8.3239346, 10.0787172, 8.3357964, 10.0966883, -1.2128623, 1.1891067
7: -14.0343885, -12.1280365, -14.0345125, -12.1250725, -1.0620213, 1.0603027
8: -6.0779772, -4.6350412, -6.0889006, -4.6395168, -0.7608241, 0.7807784
9: -10.8292656, -8.5230637, -10.8411627, -8.5272207, -1.7082047, 1.7373524

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B2_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6780762, upper bound: 0.6611114
time: 4.77 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6810680, upper bound: 0.6716740
time: 4.27 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6411318, 0.1220149, -1.6476203, 0.1169798, -1.4375062, 1.4569240
1: -17.9519424, -15.6439171, -17.9555779, -15.6547756, -1.4359641, 1.4324558
2: -6.5822186, -4.5014949, -6.5813909, -4.4958820, -1.3560514, 1.3508816
3: -14.0145350, -12.1214981, -14.0223885, -12.1249704, -1.1990042, 1.1961646
4: -5.5825682, -3.6900628, -5.5550489, -3.7041898, -1.6687889, 1.6993642
5: -7.0408154, -5.5918417, -7.0528660, -5.5918775, -0.9938107, 0.9701409
6: 8.3206587, 10.1027184, 8.3304272, 10.1245461, -1.2391624, 1.1794627
7: -14.0349464, -12.1247444, -14.0345459, -12.1235714, -1.0633898, 1.0626860
8: -6.0926561, -4.6411138, -6.0937371, -4.6519098, -0.7577105, 0.7822570
9: -10.8387680, -8.5272064, -10.8333712, -8.5326481, -1.7244334, 1.7292628

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6779142, upper bound: 0.6677655
time: 4.70 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6817549, upper bound: 0.6800989
time: 4.42 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6415184, 0.1228378, -1.6443675, 0.1178832, -1.4485636, 1.4532604
1: -17.9518929, -15.6355352, -17.9517345, -15.6380920, -1.4518857, 1.4383354
2: -6.5796952, -4.5011053, -6.5747538, -4.5017309, -1.3538876, 1.3516049
3: -14.0147734, -12.1221828, -14.0143270, -12.1277294, -1.2047710, 1.1923261
4: -5.6056080, -3.6895614, -5.6041737, -3.6895466, -1.7030220, 1.6937766
5: -7.0413523, -5.5947528, -7.0410724, -5.5999684, -1.0027204, 0.9618514
6: 8.3229103, 10.1027412, 8.3357878, 10.1097345, -1.2442555, 1.1837971
7: -14.0351696, -12.1245842, -14.0349083, -12.1232338, -1.0636673, 1.0637186
8: -6.0938253, -4.6351709, -6.0971365, -4.6395159, -0.7657380, 0.7927783
9: -10.8462067, -8.5269384, -10.8493528, -8.5272198, -1.7247753, 1.7483320

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A2_B2_A2_B2_A1

### Relational analysis result of IS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6788682, upper bound: 0.6659749
time: 5.67 seconds

## Relational analysis of IS_A2_B2_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6821523, upper bound: 0.6768597
time: 4.45 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.90 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6833010, upper bound: 0.6757284
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6863915, upper bound: 0.6852173
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6837531, upper bound: 0.6734059
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6865726, upper bound: 0.6819523
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6839958, upper bound: 0.6807937
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6876894, upper bound: 0.6911633
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6845709, upper bound: 0.6785324
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6879671, upper bound: 0.6879667
IS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6734597, upper bound: 0.6676928
IS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6778428, upper bound: 0.6803934
IS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6741639, upper bound: 0.6661273
IS_A1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6781471, upper bound: 0.6770879
IS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6741140, upper bound: 0.6723569
IS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6787417, upper bound: 0.6853621
IS_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6749432, upper bound: 0.6708266
IS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6791011, upper bound: 0.6821525
IS_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6771497, upper bound: 0.6651164
IS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6807602, upper bound: 0.6764525
IS_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6780762, upper bound: 0.6630160
IS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6810679, upper bound: 0.6731594
IS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6779142, upper bound: 0.6707123
IS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6817549, upper bound: 0.6823214
IS_A2_B1_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6788682, upper bound: 0.6686686
IS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6821522, upper bound: 0.6791016
IS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6771497, upper bound: 0.6630655
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6807605, upper bound: 0.6749819
IS_A2_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6780762, upper bound: 0.6611114
IS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6810680, upper bound: 0.6716740
IS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6779142, upper bound: 0.6677655
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6817549, upper bound: 0.6800989
IS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6788682, upper bound: 0.6659749
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.90
Output dim: 6, lower bound: -0.6821523, upper bound: 0.6768597

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6386757, 0.0811823, -1.6468296, 0.0881809, -1.4021044, 1.4004779
1: -17.9010754, -15.6727104, -17.9183197, -15.6935043, -1.2516713, 1.2027092
2: -6.5192013, -4.5903435, -6.5233417, -4.5303407, -1.2670343, 1.2187362
3: -13.9073620, -12.1626358, -13.9542866, -12.1378136, -1.1059318, 1.1299362
4: -5.5042858, -3.7295842, -5.5214734, -3.7412670, -1.5768538, 1.6152101
5: -7.0152311, -5.6015606, -7.0412512, -5.6092954, -0.8754745, 0.9186776
6: 8.3182087, 10.0109119, 8.3086824, 10.0369034, -1.1589150, 1.1595268
7: -14.0020466, -12.2630873, -14.0080576, -12.1975355, -0.9248381, 0.8867621
8: -6.0343342, -4.6461544, -6.0679231, -4.6710196, -0.7150000, 0.7694266
9: -10.7455788, -8.5082035, -10.7776136, -8.5160847, -1.6159902, 1.6593490

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6744291, upper bound: 0.6697397
time: 4.86 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6750913, upper bound: 0.6690405
time: 6.90 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6421474, 0.1063525, -1.6495898, 0.1011699, -1.4086118, 1.4205785
1: -17.9397621, -15.6853428, -17.9550114, -15.6824636, -1.2399850, 1.3437095
2: -6.5259781, -4.5056095, -6.5282588, -4.4859667, -1.3321433, 1.2462463
3: -13.9632072, -12.1464043, -13.9835176, -12.1377068, -1.1073856, 1.1718335
4: -5.5411439, -3.7336762, -5.5428834, -3.7344153, -1.6047068, 1.6699514
5: -7.0372043, -5.6149516, -7.0567713, -5.6093178, -0.8999646, 0.9373591
6: 8.3110418, 10.0074577, 8.3010368, 10.0421400, -1.1974061, 1.1909447
7: -14.0077419, -12.1559505, -14.0079985, -12.1395826, -1.0443704, 0.8749313
8: -6.0639501, -4.6604376, -6.0863285, -4.6710777, -0.7446404, 0.7778957
9: -10.7960339, -8.5060806, -10.8090706, -8.5150480, -1.6764779, 1.7166505

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6775643, upper bound: 0.6787952
time: 4.81 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6782921, upper bound: 0.6787079
time: 4.18 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6390841, 0.0814969, -1.6434582, 0.0887672, -1.4138503, 1.3961902
1: -17.9010220, -15.6557398, -17.9206524, -15.6790457, -1.2523856, 1.2176125
2: -6.5171227, -4.5898533, -6.5167027, -4.5360818, -1.2660902, 1.2194829
3: -13.9076929, -12.1633215, -13.9464073, -12.1405668, -1.1112385, 1.1267743
4: -5.5276995, -3.7290800, -5.5720186, -3.7266548, -1.6119480, 1.6105766
5: -7.0158291, -5.6023831, -7.0315189, -5.6178370, -0.8848338, 0.9103879
6: 8.3195457, 10.0109425, 8.3140020, 10.0221996, -1.1605067, 1.1680536
7: -14.0025253, -12.2629452, -14.0084200, -12.1972256, -0.9253759, 0.8874402
8: -6.0356169, -4.6402116, -6.0711164, -4.6586261, -0.7183559, 0.7825481
9: -10.7520742, -8.5078545, -10.7941017, -8.5105381, -1.6165257, 1.6817641

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A1_B2_A1_B1

### Relational analysis result of IS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6748641, upper bound: 0.6673845
time: 4.72 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6755911, upper bound: 0.6667142
time: 4.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6425309, 0.1071779, -1.6461245, 0.1019552, -1.4196153, 1.4173603
1: -17.9397087, -15.6783447, -17.9509354, -15.6673098, -1.2321942, 1.3490033
2: -6.5234528, -4.5052109, -6.5216246, -4.4917383, -1.3298631, 1.2467933
3: -13.9634390, -12.1470890, -13.9755421, -12.1404610, -1.1123357, 1.1678619
4: -5.5636339, -3.7331986, -5.5917549, -3.7197907, -1.6384907, 1.6641135
5: -7.0377402, -5.6179757, -7.0447111, -5.6178603, -0.9085686, 0.9290807
6: 8.3124704, 10.0074797, 8.3054438, 10.0273447, -1.1973372, 1.1956239
7: -14.0079613, -12.1557837, -14.0083647, -12.1392326, -1.0446792, 0.8760641
8: -6.0651374, -4.6544924, -6.0895996, -4.6586838, -0.7525637, 0.7881839
9: -10.8035269, -8.5057840, -10.8249931, -8.5095730, -1.6772413, 1.7299051

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6777648, upper bound: 0.6755031
time: 4.46 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6784846, upper bound: 0.6754394
time: 3.96 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6399013, 0.0765308, -1.6478738, 0.0881814, -1.4041991, 1.4043837
1: -17.8929825, -15.6486349, -17.9183178, -15.6806068, -1.2650251, 1.2214432
2: -6.5232534, -4.5902963, -6.5254211, -4.5302896, -1.2716341, 1.2218971
3: -13.9168139, -12.1396446, -13.9542933, -12.1267910, -1.1202130, 1.1310382
4: -5.5389905, -3.7169826, -5.5383129, -3.7398958, -1.5799932, 1.6400466
5: -7.0134401, -5.5876784, -7.0422692, -5.6009107, -0.9026604, 0.9125283
6: 8.3152199, 10.0366192, 8.3086729, 10.0508976, -1.1898279, 1.1544194
7: -14.0028276, -12.2596130, -14.0084524, -12.1956635, -0.9260440, 0.8899183
8: -6.0501523, -4.6462831, -6.0762081, -4.6710176, -0.7197064, 0.7816889
9: -10.7599735, -8.5120468, -10.7854729, -8.5160818, -1.6317458, 1.6695633

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A2_B1_A1_B1

### Relational analysis result of IS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6746455, upper bound: 0.6744749
time: 5.36 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_B2

### Relational analysis result of IS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6757513, upper bound: 0.6725506
time: 4.20 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6431763, 0.1017161, -1.6506268, 0.1011700, -1.4108739, 1.4245257
1: -17.9316673, -15.6614075, -17.9550095, -15.6692362, -1.2536354, 1.3612890
2: -6.5300713, -4.5055599, -6.5303688, -4.4859128, -1.3370252, 1.2493749
3: -13.9726553, -12.1234198, -13.9835262, -12.1266861, -1.1216912, 1.1729169
4: -5.5758476, -3.7209413, -5.5597205, -3.7331614, -1.6080151, 1.6934381
5: -7.0357518, -5.6010656, -7.0577726, -5.6009355, -0.9270182, 0.9311171
6: 8.3081160, 10.0335340, 8.3010254, 10.0560894, -1.2285161, 1.1856246
7: -14.0085201, -12.1524734, -14.0083942, -12.1377153, -1.0457296, 0.8779852
8: -6.0797658, -4.6605659, -6.0945873, -4.6710763, -0.7492309, 0.7900164
9: -10.8122482, -8.5099468, -10.8170109, -8.5150442, -1.6926327, 1.7276964

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A2_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6783936, upper bound: 0.6845654
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6795599, upper bound: 0.6830498
time: 3.94 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6403110, 0.0768436, -1.6444812, 0.0887667, -1.4160352, 1.4000778
1: -17.8929291, -15.6316347, -17.9206543, -15.6659374, -1.2657061, 1.2362204
2: -6.5211754, -4.5898056, -6.5187831, -4.5360289, -1.2706897, 1.2226300
3: -13.9171438, -12.1403303, -13.9464169, -12.1295471, -1.1255169, 1.1278768
4: -5.5624027, -3.7164433, -5.5888586, -3.7252655, -1.6150379, 1.6354489
5: -7.0140562, -5.5885015, -7.0325203, -5.6094561, -0.9121044, 0.9041632
6: 8.3165655, 10.0366497, 8.3139915, 10.0361776, -1.1913476, 1.1630652
7: -14.0033045, -12.2594719, -14.0088158, -12.1953497, -0.9265816, 0.8905962
8: -6.0514226, -4.6403408, -6.0794115, -4.6586256, -0.7231857, 0.7947171
9: -10.7665319, -8.5116997, -10.8019581, -8.5105362, -1.6322489, 1.6919317

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A2_B2_A1_B1

### Relational analysis result of IS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6751570, upper bound: 0.6721559
time: 4.58 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_B2

### Relational analysis result of IS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6763347, upper bound: 0.6703052
time: 4.60 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6435617, 0.1025395, -1.6471447, 0.1019540, -1.4219480, 1.4213071
1: -17.9316177, -15.6541500, -17.9509354, -15.6538916, -1.2457848, 1.3665187
2: -6.5275507, -4.5051627, -6.5237322, -4.4916859, -1.3347344, 1.2499084
3: -13.9728928, -12.1241035, -13.9755459, -12.1294403, -1.1266413, 1.1689458
4: -5.5983386, -3.7204294, -5.6085930, -3.7185225, -1.6417904, 1.6877041
5: -7.0363064, -5.6040931, -7.0456662, -5.6094799, -0.9356811, 0.9228091
6: 8.3095551, 10.0335541, 8.3054342, 10.0412779, -1.2284355, 1.1905575
7: -14.0087414, -12.1523075, -14.0087595, -12.1373653, -1.0460434, 0.8792009
8: -6.0809469, -4.6546230, -6.0978761, -4.6586847, -0.7573751, 0.8002117
9: -10.8197441, -8.5096493, -10.8329344, -8.5095749, -1.6935716, 1.7408490

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B1_A2_B2_A2_B1

### Relational analysis result of IS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6785979, upper bound: 0.6812611
time: 4.04 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6798446, upper bound: 0.6798443
time: 3.84 seconds

## BFS IS instance: IS_A1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6421474, 0.1063525, -1.6462723, 0.1149904, -1.4257183, 1.4204912
1: -17.9397621, -15.6853428, -17.9464607, -15.6696787, -1.2582803, 1.3399601
2: -6.5259781, -4.5056095, -6.5786734, -4.5030112, -1.3473649, 1.3180127
3: -13.9632072, -12.1464043, -14.0208931, -12.1360893, -1.1142383, 1.2129250
4: -5.5411439, -3.7336762, -5.5303121, -3.7063916, -1.6428127, 1.6722212
5: -7.0372043, -5.6149516, -7.0471315, -5.6003904, -0.9074669, 0.9297707
6: 8.3110418, 10.0074577, 8.3388767, 10.1108313, -1.2812304, 1.1500554
7: -14.0077419, -12.1559505, -14.0340261, -12.1319704, -1.0359614, 0.8989983
8: -6.0639501, -4.6604376, -6.0790334, -4.6520443, -0.7480223, 0.7596930
9: -10.7960339, -8.5060806, -10.8202057, -8.5328445, -1.6496582, 1.7196598

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6690286, upper bound: 0.6743749
time: 5.56 seconds

## Relational analysis of IS_A1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6696911, upper bound: 0.6609999
time: 8.05 seconds

## BFS IS instance: IS_A1_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6431763, 0.1017161, -1.6472590, 0.1149907, -1.4279771, 1.4244637
1: -17.9316673, -15.6614075, -17.9464626, -15.6563005, -1.2710419, 1.3589458
2: -6.5300713, -4.5055599, -6.5807810, -4.5029612, -1.3522458, 1.3210850
3: -13.9726553, -12.1234198, -14.0209007, -12.1250715, -1.1285443, 1.2136092
4: -5.5758476, -3.7209413, -5.5480061, -3.7051463, -1.6460876, 1.6957307
5: -7.0357518, -5.6010656, -7.0481343, -5.5919828, -0.9338665, 0.9235110
6: 8.3081160, 10.0335340, 8.3388662, 10.1239014, -1.3114376, 1.1464310
7: -14.0085201, -12.1524734, -14.0344238, -12.1301289, -1.0375960, 0.9020233
8: -6.0797658, -4.6605659, -6.0872717, -4.6520419, -0.7525191, 0.7719026
9: -10.8122482, -8.5099468, -10.8283749, -8.5328407, -1.6658134, 1.7311344

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B2_A2_B1_A2_B1

### Relational analysis result of IS_A1_B2_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6695308, upper bound: 0.6793663
time: 4.26 seconds

## Relational analysis of IS_A1_B2_A2_B1_A2_B2

### Relational analysis result of IS_A1_B2_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6705834, upper bound: 0.6764943
time: 8.93 seconds

## BFS IS instance: IS_A1_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6435617, 0.1025395, -1.6440187, 0.1157753, -1.4390717, 1.4212728
1: -17.9316177, -15.6541500, -17.9423904, -15.6397381, -1.2631907, 1.3641782
2: -6.5275507, -4.5051627, -6.5741467, -4.5087681, -1.3499751, 1.3216152
3: -13.9728928, -12.1241035, -14.0128355, -12.1278276, -1.1335063, 1.2094903
4: -5.5983386, -3.7204294, -5.5967474, -3.6904988, -1.6798515, 1.6899281
5: -7.0363064, -5.6040931, -7.0360136, -5.6000686, -0.9425542, 0.9152029
6: 8.3095551, 10.0335541, 8.3437805, 10.1090889, -1.3113604, 1.1510873
7: -14.0087414, -12.1523075, -14.0347834, -12.1297884, -1.0378728, 0.9032357
8: -6.0809469, -4.6546230, -6.0906725, -4.6396503, -0.7606628, 0.7821512
9: -10.8197441, -8.5096493, -10.8442345, -8.5273972, -1.6667485, 1.7442465

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B2_A2_B2_A2_B1

### Relational analysis result of IS_A1_B2_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6698861, upper bound: 0.6762196
time: 4.13 seconds

## Relational analysis of IS_A1_B2_A2_B2_A2_B2

### Relational analysis result of IS_A1_B2_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6709427, upper bound: 0.6732863
time: 3.88 seconds

## BFS IS instance: IS_A2_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6391573, 0.1201601, -1.6495898, 0.1011699, -1.4086008, 1.4376583
1: -17.9312134, -15.6725607, -17.9550114, -15.6824636, -1.2380481, 1.3619378
2: -6.5763679, -4.5227666, -6.5282588, -4.4859667, -1.4024453, 1.2654257
3: -14.0004768, -12.1447849, -13.9835176, -12.1377068, -1.1483736, 1.1785522
4: -5.5278130, -3.7070405, -5.5428834, -3.7344153, -1.6072302, 1.7081208
5: -7.0274725, -5.6060214, -7.0567713, -5.6093178, -0.8926980, 0.9449570
6: 8.3474369, 10.0767822, 8.3010368, 10.0421400, -1.1581609, 1.2745647
7: -14.0337811, -12.1483316, -14.0079985, -12.1395826, -1.0695884, 0.8656268
8: -6.0571980, -4.6413984, -6.0863285, -4.6710777, -0.7269740, 0.7805020
9: -10.8066177, -8.5238819, -10.8090706, -8.5150480, -1.6775670, 1.6898246

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A2_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6713862, upper bound: 0.6704150
time: 4.46 seconds

## Relational analysis of IS_A2_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6718909, upper bound: 0.6701504
time: 15.16 seconds

## BFS IS instance: IS_A2_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6395462, 0.1209851, -1.6461245, 0.1019552, -1.4194598, 1.4344311
1: -17.9311638, -15.6649132, -17.9509354, -15.6673098, -1.2302516, 1.3677418
2: -6.5738454, -4.5223780, -6.5216246, -4.4917383, -1.4001627, 1.2659564
3: -14.0007124, -12.1454706, -13.9755421, -12.1404610, -1.1532178, 1.1745825
4: -5.5502763, -3.7065506, -5.5917549, -3.7197907, -1.6409798, 1.7022805
5: -7.0280037, -5.6089201, -7.0447111, -5.6178603, -0.9012234, 0.9365027
6: 8.3490868, 10.0768013, 8.3054438, 10.0273447, -1.1581802, 1.2792482
7: -14.0340004, -12.1481762, -14.0083647, -12.1392326, -1.0698986, 0.8666542
8: -6.0583553, -4.6354566, -6.0895996, -4.6586838, -0.7346103, 0.7907922
9: -10.8140869, -8.5236006, -10.8249931, -8.5095730, -1.6783199, 1.7030811

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A2_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6716891, upper bound: 0.6671938
time: 4.57 seconds

## Relational analysis of IS_A2_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6721993, upper bound: 0.6669229
time: 4.51 seconds

## BFS IS instance: IS_A2_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6401138, 0.1155247, -1.6506268, 0.1011700, -1.4109440, 1.4416080
1: -17.9231205, -15.6485128, -17.9550095, -15.6692362, -1.2502894, 1.3794317
2: -6.5804596, -4.5227203, -6.5303688, -4.4859128, -1.4072475, 1.2685513
3: -14.0099163, -12.1218033, -13.9835262, -12.1266861, -1.1630797, 1.1796341
4: -5.5644817, -3.6929107, -5.5597205, -3.7331614, -1.6105385, 1.7315583
5: -7.0259523, -5.5921650, -7.0577726, -5.6009355, -0.9197087, 0.9387159
6: 8.3463898, 10.1008186, 8.3010254, 10.0560894, -1.1875663, 1.2691803
7: -14.0345602, -12.1448660, -14.0083942, -12.1377153, -1.0709434, 0.8688834
8: -6.0729313, -4.6415305, -6.0945873, -4.6710763, -0.7315447, 0.7927165
9: -10.8233013, -8.5277500, -10.8170109, -8.5150442, -1.6945801, 1.7008719

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A2_B1_A2_B1_A2_B1

### Relational analysis result of IS_A2_B1_A2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6718968, upper bound: 0.6761167
time: 4.46 seconds

## Relational analysis of IS_A2_B1_A2_B1_A2_B2

### Relational analysis result of IS_A2_B1_A2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6728872, upper bound: 0.6741559
time: 4.16 seconds

## BFS IS instance: IS_A2_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6405034, 0.1163489, -1.6471447, 0.1019540, -1.4218731, 1.4383802
1: -17.9230690, -15.6404428, -17.9509354, -15.6538916, -1.2424316, 1.3854084
2: -6.5779381, -4.5223355, -6.5237322, -4.4916859, -1.4049592, 1.2690692
3: -14.0101557, -12.1224909, -13.9755459, -12.1294403, -1.1679211, 1.1756673
4: -5.5869417, -3.6923926, -5.6085930, -3.7185225, -1.6442795, 1.7258229
5: -7.0265074, -5.5950661, -7.0456662, -5.6094799, -0.9282930, 0.9302319
6: 8.3480492, 10.1008396, 8.3054342, 10.0412779, -1.1875758, 1.2741270
7: -14.0347824, -12.1447077, -14.0087595, -12.1373653, -1.0712585, 0.8699939
8: -6.0740790, -4.6355863, -6.0978761, -4.6586847, -0.7393461, 0.8029128
9: -10.8307743, -8.5274668, -10.8329344, -8.5095749, -1.6955171, 1.7140250

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A2_B1_A2_B2_A2_B1

### Relational analysis result of IS_A2_B1_A2_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6722745, upper bound: 0.6728878
time: 4.36 seconds

## Relational analysis of IS_A2_B1_A2_B2_A2_B2

### Relational analysis result of IS_A2_B1_A2_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6732841, upper bound: 0.6709435
time: 7.19 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6391573, 0.1201601, -1.6462723, 0.1149904, -1.4311371, 1.4429998
1: -17.9312134, -15.6725607, -17.9464607, -15.6696787, -1.2739482, 1.3842235
2: -6.5763679, -4.5227666, -6.5786734, -4.5030112, -1.3442764, 1.2578454
3: -14.0004768, -12.1447849, -14.0208931, -12.1360893, -1.1308150, 1.1925273
4: -5.5278130, -3.7070405, -5.5303121, -3.7063916, -1.6021099, 1.6690264
5: -7.0274725, -5.6060214, -7.0471315, -5.6003904, -0.9331777, 0.9743497
6: 8.3474369, 10.0767822, 8.3388767, 10.1108313, -1.1816051, 1.1755900
7: -14.0337811, -12.1483316, -14.0340261, -12.1319704, -1.0601151, 0.8885856
8: -6.0571980, -4.6413984, -6.0790334, -4.6520443, -0.7315181, 0.7638218
9: -10.8066177, -8.5238819, -10.8202057, -8.5328445, -1.6730919, 1.7135239

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6713862, upper bound: 0.6689797
time: 8.15 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6718910, upper bound: 0.6682968
time: 6.27 seconds

## BFS IS instance: IS_A2_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6395462, 0.1209851, -1.6430496, 0.1157751, -1.4420147, 1.4398022
1: -17.9311638, -15.6649132, -17.9423866, -15.6533108, -1.2660046, 1.3899431
2: -6.5738454, -4.5223780, -6.5720382, -4.5088191, -1.3420076, 1.2583790
3: -14.0007124, -12.1454706, -14.0128288, -12.1388454, -1.1356301, 1.1883774
4: -5.5502763, -3.7065506, -5.5790539, -3.6917579, -1.6358671, 1.6631484
5: -7.0280037, -5.6089201, -7.0350571, -5.6084733, -0.9417870, 0.9659467
6: 8.3490868, 10.0768013, 8.3437891, 10.0960379, -1.1818466, 1.1803045
7: -14.0340004, -12.1481762, -14.0343885, -12.1316299, -1.0604022, 0.8896091
8: -6.0583553, -4.6354566, -6.0824466, -4.6396489, -0.7392220, 0.7743095
9: -10.8140869, -8.5236006, -10.8360662, -8.5273991, -1.6738553, 1.7267938

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A2_B2_A1_B2_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6716892, upper bound: 0.6657446
time: 6.10 seconds

## Relational analysis of IS_A2_B2_A1_B2_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6721995, upper bound: 0.6650166
time: 4.43 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 16.33 seconds
IS_A1_B1_A1_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6744291, upper bound: 0.6697397
IS_A1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6750913, upper bound: 0.6690405
IS_A1_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6775643, upper bound: 0.6787952
IS_A1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6782921, upper bound: 0.6787079
IS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6748641, upper bound: 0.6673845
IS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6755911, upper bound: 0.6667142
IS_A1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6777648, upper bound: 0.6755031
IS_A1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6784846, upper bound: 0.6754394
IS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6746455, upper bound: 0.6744749
IS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6757513, upper bound: 0.6725506
IS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6783936, upper bound: 0.6845654
IS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6795599, upper bound: 0.6830498
IS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6751570, upper bound: 0.6721559
IS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6763347, upper bound: 0.6703052
IS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6785979, upper bound: 0.6812611
IS_A1_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6798446, upper bound: 0.6798443
IS_A1_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6690286, upper bound: 0.6743749
IS_A1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6696911, upper bound: 0.6609999
IS_A1_B2_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6695308, upper bound: 0.6793663
IS_A1_B2_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6705834, upper bound: 0.6764943
IS_A1_B2_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6698861, upper bound: 0.6762196
IS_A1_B2_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6709427, upper bound: 0.6732863
IS_A2_B1_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6713862, upper bound: 0.6704150
IS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6718909, upper bound: 0.6701504
IS_A2_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6716891, upper bound: 0.6671938
IS_A2_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6721993, upper bound: 0.6669229
IS_A2_B1_A2_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6718968, upper bound: 0.6761167
IS_A2_B1_A2_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6728872, upper bound: 0.6741559
IS_A2_B1_A2_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6722745, upper bound: 0.6728878
IS_A2_B1_A2_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6732841, upper bound: 0.6709435
IS_A2_B2_A1_B1_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6713862, upper bound: 0.6689797
IS_A2_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6718910, upper bound: 0.6682968
IS_A2_B2_A1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6716892, upper bound: 0.6657446
IS_A2_B2_A1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 16.33
Output dim: 6, lower bound: -0.6721995, upper bound: 0.6650166
IS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 16.33
Output dim: 6, lower bound: -0.6817549, upper bound: 0.6800989
IS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 16.33
Output dim: 6, lower bound: -0.6821523, upper bound: 0.6768597
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.2817935943603516
rel_dist={6: [-0.7326686905779951, 0.7326686039477615]}

## Binary Search with IS_dual_ind Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2416.70 seconds
