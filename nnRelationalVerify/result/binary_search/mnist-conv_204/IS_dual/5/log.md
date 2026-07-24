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
execution time: IAR + LP analysis = 15.12 + 32.09 = 47.21 seconds
status: Status.ADV_EXAMPLE


# Binary Search by BASE starts (time budget: 3552.79 seconds, max iter: 100)

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
Binary search time: 209.48 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.0078125


# Individual Split (IS_dual) starts
Time budget: 3343.30 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.01 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: B, layer: 3, pos: 2130
type: A, layer: 3, pos: 2130
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0936775, upper bound: 1.0987745
time: 3.29 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0936775, upper bound: 1.0936774
time: 3.50 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 7.10 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 7.10
Output dim: 6, lower bound: -1.0936775, upper bound: 1.0987745
IS_B2, status: Status.UNKNOWN, split count: 1, time: 7.10
Output dim: 6, lower bound: -1.0936775, upper bound: 1.0936774

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -1.6561922, 0.1125274, -1.6506550, 0.1124235, -1.6829329, 1.6750298
1: -17.9640560, -15.6267805, -17.9605484, -15.6270342, -1.9597316, 1.9554849
2: -6.5349898, -4.4621940, -6.5349741, -4.4834495, -1.7481227, 1.7719665
3: -13.9784403, -12.1074400, -13.9778080, -12.1164856, -1.5706391, 1.6009035
4: -5.6369295, -3.7163720, -5.6252713, -3.7163892, -1.9205403, 1.9088993
5: -7.0538297, -5.5888457, -7.0521522, -5.5889454, -1.3086953, 1.3117192
6: 8.2564125, 10.0458202, 8.2744751, 10.0458050, -1.6146502, 1.6015844
7: -14.0097389, -12.1174345, -14.0097361, -12.1294889, -1.4694927, 1.4759142
8: -6.1156301, -4.6512346, -6.1093316, -4.6512423, -1.1304324, 1.1277353
9: -10.8449526, -8.5048056, -10.8447342, -8.5088797, -2.1446929, 2.1462402

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2130
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2130

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0884662
time: 3.52 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0884661
time: 3.75 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -1.6545560, 0.1124785, -1.6475005, 0.1262470, -1.7041864, 1.6745195
1: -17.9626312, -15.6268787, -17.9519997, -15.6137323, -1.9757223, 1.9853816
2: -6.5349855, -4.4676447, -6.5853863, -4.5002308, -1.7628379, 1.8386312
3: -13.9781818, -12.1083164, -14.0151386, -12.1148701, -1.6312375, 1.6408954
4: -5.6334400, -3.7163763, -5.6131539, -3.6883640, -1.9450760, 1.8967776
5: -7.0524797, -5.5888901, -7.0425291, -5.5800605, -1.3117352, 1.3461246
6: 8.2663641, 10.0458136, 8.3123417, 10.1139135, -1.6856112, 1.5765390
7: -14.0097370, -12.1193743, -14.0357504, -12.1219559, -1.4616511, 1.4979756
8: -6.1131978, -4.6512384, -6.1018839, -4.6322112, -1.1296992, 1.1163242
9: -10.8448553, -8.5076971, -10.8561497, -8.5267410, -2.1463761, 2.1452470

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2130
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2130

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0936776
time: 3.53 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0936794
time: 3.36 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 12.51 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 12.51
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0884662
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 12.51
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0884661
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 12.51
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0936776
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 12.51
Output dim: 6, lower bound: -1.0884658, upper bound: 1.0936794

## BFS IS instance: IS_B1_A1

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

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0873176
time: 3.52 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0755620
time: 3.80 seconds

## BFS IS instance: IS_B1_A2

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

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0873193
time: 3.78 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0755620
time: 3.82 seconds

## BFS IS instance: IS_B2_A1

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

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0768305, upper bound: 1.0702284
time: 4.70 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0702264
time: 6.50 seconds

## BFS IS instance: IS_B2_A2

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

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0768305, upper bound: 1.0651272
time: 4.20 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0651276
time: 3.77 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.60 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 13.60
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0873176
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 13.60
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0755620
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 13.60
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0873193
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 13.60
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0755620
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 13.60
Output dim: 6, lower bound: -1.0768305, upper bound: 1.0702284
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 13.60
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0702264
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 13.60
Output dim: 6, lower bound: -1.0768305, upper bound: 1.0651272
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 13.60
Output dim: 6, lower bound: -1.0651270, upper bound: 1.0651276

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -1.6495955, 0.1085103, -1.6506550, 0.1124235, -1.6724391, 1.6694107
1: -17.9605465, -15.6395464, -17.9605484, -15.6270342, -1.9474125, 1.9359534
2: -6.5346708, -4.4855671, -6.5349741, -4.4834495, -1.7431207, 1.7412672
3: -13.9729347, -12.1166830, -13.9778080, -12.1164856, -1.5520530, 1.5577888
4: -5.6134143, -3.7176580, -5.6252713, -3.7163892, -1.8970251, 1.9076133
5: -7.0508256, -5.5946903, -7.0521522, -5.5889454, -1.2983060, 1.2929053
6: 8.2793570, 10.0396023, 8.2744751, 10.0458050, -1.5924191, 1.5864391
7: -14.0071535, -12.1302347, -14.0097361, -12.1294889, -1.4623635, 1.4675238
8: -6.0897551, -4.6513085, -6.1093316, -4.6512423, -1.1066685, 1.1276407
9: -10.8406801, -8.5175896, -10.8447342, -8.5088797, -2.1328321, 2.1274624

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0819898
time: 4.09 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0703562, upper bound: 1.0820438
time: 4.02 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -1.6776519, 0.1100102, -1.6503273, 0.1117609, -1.7013040, 1.6683168
1: -17.9867172, -15.6479759, -17.9605465, -15.6302004, -1.9721322, 1.9344406
2: -6.5353975, -4.4761448, -6.5348954, -4.4838986, -1.7594886, 1.7457705
3: -13.9640341, -12.0983925, -13.9756365, -12.1165295, -1.5535164, 1.5759449
4: -5.6055136, -3.6839080, -5.6217756, -3.7167192, -1.8887944, 1.9378676
5: -7.0707974, -5.5970683, -7.0518341, -5.5901289, -1.3226721, 1.3088534
6: 8.2517538, 10.0169201, 8.2757053, 10.0419273, -1.6372523, 1.5779152
7: -13.9917107, -12.1216211, -14.0070219, -12.1296864, -1.4583838, 1.4861395
8: -6.0912781, -4.5999527, -6.1056881, -4.6512809, -1.1306720, 1.1858464
9: -10.8720083, -8.5257854, -10.8438025, -8.5111771, -2.1688185, 2.1663418

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0703077
time: 4.18 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0703562, upper bound: 1.0703566
time: 3.85 seconds

## BFS IS instance: IS_B1_A2_A1

### Backsubstitution after applying IS history:
0: -1.6464379, 0.1223224, -1.6506550, 0.1124235, -1.6725936, 1.6866074
1: -17.9519958, -15.6262808, -17.9605484, -15.6270342, -1.9439435, 1.9533508
2: -6.5850840, -4.5023246, -6.5349741, -4.4834495, -1.8134651, 1.7569189
3: -14.0102634, -12.1150637, -13.9778080, -12.1164856, -1.5929089, 1.5644951
4: -5.6012630, -3.6896396, -5.6252713, -3.7163892, -1.8848739, 1.9356318
5: -7.0412030, -5.5858030, -7.0521522, -5.5889454, -1.2906728, 1.2993934
6: 8.3170557, 10.1077023, 8.2744751, 10.0458050, -1.5513940, 1.6687369
7: -14.0331659, -12.1226940, -14.0097361, -12.1294889, -1.4874775, 1.4591632
8: -6.0823116, -4.6322727, -6.1093316, -4.6512423, -1.0889573, 1.1302934
9: -10.8520432, -8.5354223, -10.8447342, -8.5088797, -2.1353059, 2.1006889

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0512660, upper bound: 1.0740079
time: 4.12 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0557637, upper bound: 1.0740235
time: 4.36 seconds

## BFS IS instance: IS_B1_A2_A2

### Backsubstitution after applying IS history:
0: -1.6742057, 0.1238362, -1.6503273, 0.1117609, -1.7011151, 1.6855636
1: -17.9781647, -15.6348305, -17.9605465, -15.6302004, -1.9686713, 1.9514909
2: -6.5858092, -4.4928999, -6.5348954, -4.4838986, -1.8298140, 1.7613745
3: -14.0013285, -12.0967751, -13.9756365, -12.1165295, -1.5944867, 1.5826020
4: -5.5931997, -3.6559031, -5.6217756, -3.7167192, -1.8764806, 1.9658725
5: -7.0611372, -5.5880609, -7.0518341, -5.5901289, -1.3151362, 1.3156528
6: 8.2899466, 10.0849972, 8.2757053, 10.0419273, -1.5961103, 1.6602135
7: -14.0177317, -12.1141577, -14.0070219, -12.1296864, -1.4835093, 1.4780953
8: -6.0839720, -4.5808887, -6.1056881, -4.6512809, -1.1129830, 1.1885477
9: -10.8831568, -8.5435457, -10.8438025, -8.5111771, -2.1709471, 2.1395717

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0512660, upper bound: 1.0622385
time: 4.00 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0557637, upper bound: 1.0622463
time: 4.03 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6506550, 0.1124235, -1.6464379, 0.1223224, -1.6866074, 1.6725936
1: -17.9605484, -15.6270342, -17.9519958, -15.6262808, -1.9533510, 1.9439435
2: -6.5349741, -4.4834495, -6.5850840, -4.5023246, -1.7569189, 1.8134651
3: -13.9778080, -12.1164856, -14.0102634, -12.1150637, -1.5644951, 1.5929089
4: -5.6252713, -3.7163892, -5.6012630, -3.6896396, -1.9356318, 1.8848739
5: -7.0521522, -5.5889454, -7.0412030, -5.5858030, -1.2993934, 1.2906728
6: 8.2744751, 10.0458050, 8.3170557, 10.1077023, -1.6687369, 1.5513940
7: -14.0097361, -12.1294889, -14.0331659, -12.1226940, -1.4591632, 1.4874775
8: -6.1093316, -4.6512423, -6.0823116, -4.6322727, -1.1302934, 1.0889574
9: -10.8447342, -8.5088797, -10.8520432, -8.5354223, -2.1006889, 2.1353068

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0740062, upper bound: 1.0512662
time: 3.96 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0740240, upper bound: 1.0557639
time: 3.65 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6503273, 0.1117609, -1.6742057, 0.1238362, -1.6855636, 1.7011151
1: -17.9605465, -15.6302004, -17.9781647, -15.6348305, -1.9514909, 1.9686713
2: -6.5348954, -4.4838986, -6.5858092, -4.4928999, -1.7613745, 1.8298140
3: -13.9756365, -12.1165295, -14.0013285, -12.0967751, -1.5826020, 1.5944867
4: -5.6217756, -3.7167192, -5.5931997, -3.6559031, -1.9658725, 1.8764806
5: -7.0518341, -5.5901289, -7.0611372, -5.5880609, -1.3156528, 1.3151362
6: 8.2757053, 10.0419273, 8.2899466, 10.0849972, -1.6602135, 1.5961103
7: -14.0070219, -12.1296864, -14.0177317, -12.1141577, -1.4780953, 1.4835093
8: -6.1056881, -4.6512809, -6.0839720, -4.5808887, -1.1885478, 1.1129830
9: -10.8438025, -8.5111771, -10.8831568, -8.5435457, -2.1395721, 2.1709471

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0622387, upper bound: 1.0512661
time: 3.86 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0622445, upper bound: 1.0557637
time: 3.67 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6475005, 0.1262470, -1.6464379, 0.1223224, -1.6923466, 1.6953435
1: -17.9519997, -15.6137323, -17.9519958, -15.6262808, -1.9747438, 1.9858382
2: -6.5853863, -4.5002308, -6.5850840, -4.5023246, -1.7618747, 1.7637138
3: -14.0151386, -12.1148701, -14.0102634, -12.1150637, -1.5965447, 1.5906720
4: -5.6131539, -3.6883640, -5.6012630, -3.6896396, -1.9235144, 1.9128990
5: -7.0425291, -5.5800605, -7.0412030, -5.5858030, -1.3349309, 1.3404710
6: 8.3123417, 10.1139135, 8.3170557, 10.1077023, -1.5792663, 1.5851865
7: -14.0357504, -12.1219559, -14.0331659, -12.1226940, -1.4834874, 1.4782858
8: -6.1018839, -4.6322112, -6.0823116, -4.6322727, -1.1149876, 1.0940125
9: -10.8561497, -8.5267410, -10.8520432, -8.5354223, -2.1287079, 2.1340933

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0675086, upper bound: 1.0454813
time: 4.47 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0675624, upper bound: 1.0512296
time: 3.64 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6471677, 0.1255836, -1.6742057, 0.1238362, -1.6913033, 1.7238669
1: -17.9519958, -15.6169100, -17.9781647, -15.6348305, -1.9734049, 2.0106738
2: -6.5853071, -4.5006742, -6.5858092, -4.4928999, -1.7661953, 1.7800646
3: -14.0129662, -12.1149130, -14.0013285, -12.0967751, -1.6148891, 1.5921974
4: -5.6096506, -3.6886947, -5.5931997, -3.6559031, -1.9537475, 1.9045050
5: -7.0422115, -5.5812435, -7.0611372, -5.5880609, -1.3509264, 1.3647885
6: 8.3135452, 10.1100368, 8.2899466, 10.0849972, -1.5706797, 1.6294975
7: -14.0330324, -12.1221523, -14.0177317, -12.1141577, -1.5024185, 1.4743292
8: -6.0982385, -4.6322479, -6.0839720, -4.5808887, -1.1731424, 1.1177490
9: -10.8552046, -8.5290327, -10.8831568, -8.5435457, -2.1675811, 2.1699915

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0556912, upper bound: 1.0454833
time: 3.92 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0557637, upper bound: 1.0512296
time: 3.49 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 13.04 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0819898
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0703562, upper bound: 1.0820438
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0703077
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0703562, upper bound: 1.0703566
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0512660, upper bound: 1.0740079
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0557637, upper bound: 1.0740235
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0512660, upper bound: 1.0622385
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0557637, upper bound: 1.0622463
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0740062, upper bound: 1.0512662
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0740240, upper bound: 1.0557639
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0622387, upper bound: 1.0512661
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0622445, upper bound: 1.0557637
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0675086, upper bound: 1.0454813
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0675624, upper bound: 1.0512296
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0556912, upper bound: 1.0454833
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 13.04
Output dim: 6, lower bound: -1.0557637, upper bound: 1.0512296

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6485716, 0.1085088, -1.6440289, 0.1170566, -1.6712370, 1.6576471
1: -17.9605465, -15.6460991, -17.9686394, -15.6648798, -1.9024744, 1.9330926
2: -6.5337744, -4.4856172, -6.5292134, -4.4837713, -1.7417784, 1.7348261
3: -13.9729290, -12.1206341, -13.9683418, -12.1419897, -1.5333595, 1.5490131
4: -5.6072845, -3.7181706, -5.5857277, -3.7300534, -1.8772311, 1.8675570
5: -7.0504222, -5.5984144, -7.0527544, -5.6105976, -1.2650838, 1.2736309
6: 8.2793627, 10.0337639, 8.2773838, 10.0093632, -1.5530381, 1.5675702
7: -14.0069876, -12.1311550, -14.0086727, -12.1354914, -1.4562540, 1.4653509
8: -6.0860777, -4.6513085, -6.0861721, -4.6511135, -1.1005231, 1.1023533
9: -10.8368025, -8.5175915, -10.8196745, -8.5050039, -2.1297140, 2.0967627

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0819898
time: 3.95 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0819898
time: 4.16 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -1.6495955, 0.1085103, -1.6450552, 0.1124195, -1.6724215, 1.6591086
1: -17.9605465, -15.6395464, -17.9605503, -15.6407909, -1.9252291, 1.9359212
2: -6.5346708, -4.4855671, -6.5333314, -4.4837227, -1.7432976, 1.7397633
3: -13.9729347, -12.1166830, -13.9777908, -12.1190071, -1.5424562, 1.5577860
4: -5.6134143, -3.7176580, -5.6204309, -3.7172384, -1.8961759, 1.9027729
5: -7.0508256, -5.5946903, -7.0514894, -5.5967188, -1.2594895, 1.2902367
6: 8.2793570, 10.0396023, 8.2744904, 10.0354328, -1.5516839, 1.5864339
7: -14.0071535, -12.1302347, -14.0094528, -12.1319809, -1.4591632, 1.4662454
8: -6.0897551, -4.6513085, -6.1021070, -4.6512437, -1.1066310, 1.1095867
9: -10.8406801, -8.5175896, -10.8361607, -8.5088787, -2.1328330, 2.1131568

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B1_A1_A1_B2_B1

### Relational analysis result of IS_B1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0703562, upper bound: 1.0820436
time: 3.87 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2

### Relational analysis result of IS_B1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0703562, upper bound: 1.0820438
time: 4.15 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -1.6766319, 0.1100085, -1.6436982, 0.1163945, -1.7001100, 1.6565547
1: -17.9867172, -15.6544209, -17.9686394, -15.6680336, -1.9271712, 1.9314516
2: -6.5345006, -4.4761944, -6.5291390, -4.4842200, -1.7581558, 1.7393317
3: -13.9640303, -12.1023426, -13.9661732, -12.1420336, -1.5348201, 1.5671697
4: -5.5993857, -3.6844392, -5.5822315, -3.7303596, -1.8690262, 1.8977923
5: -7.0704103, -5.6007915, -7.0524511, -5.6117826, -1.2894542, 1.2895894
6: 8.2517605, 10.0110798, 8.2786198, 10.0054932, -1.5978637, 1.5590429
7: -13.9915466, -12.1225433, -14.0059566, -12.1356945, -1.4522758, 1.4839809
8: -6.0875735, -4.5999537, -6.0825377, -4.6511531, -1.1245334, 1.1605525
9: -10.8681459, -8.5257854, -10.8187342, -8.5073051, -2.1657882, 2.1356535

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_B1_A1_A2_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0266211, upper bound: 1.0491982
time: 3.95 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0497349, upper bound: 1.0567052
time: 4.07 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -1.6776519, 0.1100102, -1.6447271, 0.1117580, -1.7012868, 1.6580462
1: -17.9867172, -15.6479759, -17.9605484, -15.6439562, -1.9499383, 1.9344084
2: -6.5353975, -4.4761448, -6.5332546, -4.4841700, -1.7596645, 1.7442665
3: -13.9640341, -12.0983925, -13.9756250, -12.1190510, -1.5439563, 1.5759420
4: -5.6055136, -3.6839080, -5.6169357, -3.7175634, -1.8879502, 1.9330277
5: -7.0707974, -5.5970683, -7.0511684, -5.5979013, -1.2839420, 1.3061864
6: 8.2517538, 10.0169201, 8.2757215, 10.0315580, -1.5966239, 1.5779104
7: -13.9917107, -12.1216211, -14.0067368, -12.1321783, -1.4551816, 1.4848249
8: -6.0912781, -4.5999527, -6.0984664, -4.6512837, -1.1306348, 1.1678064
9: -10.8720083, -8.5257854, -10.8352194, -8.5111790, -2.1688170, 2.1520534

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0703074, upper bound: 1.0633820
time: 4.14 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0703074, upper bound: 1.0703562
time: 7.83 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6454377, 0.1223217, -1.6440289, 0.1170566, -1.6713800, 1.6748428
1: -17.9519958, -15.6329288, -17.9686394, -15.6648798, -1.8990045, 1.9504659
2: -6.5841846, -4.5023756, -6.5292134, -4.4837713, -1.8121290, 1.7504783
3: -14.0102587, -12.1190176, -13.9683418, -12.1419897, -1.5742130, 1.5557184
4: -5.5949364, -3.6901467, -5.5857277, -3.7300534, -1.8648829, 1.8955810
5: -7.0408001, -5.5895252, -7.0527544, -5.6105976, -1.2574587, 1.2802823
6: 8.3170605, 10.1020603, 8.2773838, 10.0093632, -1.5120130, 1.6500731
7: -14.0330000, -12.1235991, -14.0086727, -12.1354914, -1.4813743, 1.4569173
8: -6.0786629, -4.6322737, -6.0861721, -4.6511135, -1.0828105, 1.1050062
9: -10.8480673, -8.5354233, -10.8196745, -8.5050039, -2.1320229, 2.0699887

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B1_A2_A1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0512660, upper bound: 1.0740079
time: 4.55 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0512660, upper bound: 1.0740079
time: 3.97 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6464379, 0.1223224, -1.6450552, 0.1124195, -1.6725760, 1.6763024
1: -17.9519958, -15.6262808, -17.9605503, -15.6407909, -1.9231596, 1.9533186
2: -6.5850840, -4.5023246, -6.5333314, -4.4837227, -1.8136072, 1.7554150
3: -14.0102634, -12.1150637, -13.9777908, -12.1190071, -1.5829272, 1.5644922
4: -5.6012630, -3.6896396, -5.6204309, -3.7172384, -1.8840246, 1.9307914
5: -7.0412030, -5.5858030, -7.0514894, -5.5967188, -1.2518587, 1.2967246
6: 8.3170557, 10.1077023, 8.2744904, 10.0354328, -1.5123549, 1.6687317
7: -14.0331659, -12.1226940, -14.0094528, -12.1319809, -1.4842768, 1.4580605
8: -6.0823116, -4.6322727, -6.1021070, -4.6512437, -1.0889201, 1.1121469
9: -10.8520432, -8.5354223, -10.8361607, -8.5088787, -2.1353068, 2.0863833

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B1_A2_A1_B2_B1

### Relational analysis result of IS_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0557637, upper bound: 1.0740257
time: 4.99 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2

### Relational analysis result of IS_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0557637, upper bound: 1.0740235
time: 4.26 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6732104, 0.1238357, -1.6436982, 0.1163945, -1.6999102, 1.6738000
1: -17.9781685, -15.6414309, -17.9686394, -15.6680336, -1.9237099, 1.9484611
2: -6.5849137, -4.4929523, -6.5291390, -4.4842200, -1.8284855, 1.7549367
3: -14.0013247, -12.1007271, -13.9661732, -12.1420336, -1.5757885, 1.5738273
4: -5.5868731, -3.6564279, -5.5822315, -3.7303596, -1.8565135, 1.9258037
5: -7.0607500, -5.5917835, -7.0524511, -5.6117826, -1.2819264, 1.2965522
6: 8.2899513, 10.0793514, 8.2786198, 10.0054932, -1.5567222, 1.6415472
7: -14.0175657, -12.1150713, -14.0059566, -12.1356945, -1.4774079, 1.4759016
8: -6.0802822, -4.5808878, -6.0825377, -4.6511531, -1.1068380, 1.1632544
9: -10.8792028, -8.5435448, -10.8187342, -8.5073051, -2.1677508, 2.1088829

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_B1_A2_A2_B1_B1

### Relational analysis result of IS_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0138255, upper bound: 1.0410554
time: 4.80 seconds

## Relational analysis of IS_B1_A2_A2_B1_B2

### Relational analysis result of IS_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0366568, upper bound: 1.0487070
time: 4.91 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6742057, 0.1238362, -1.6447271, 0.1117580, -1.7010980, 1.6752720
1: -17.9781647, -15.6348305, -17.9605484, -15.6439562, -1.9478774, 1.9514587
2: -6.5858092, -4.4928999, -6.5332546, -4.4841700, -1.8299561, 1.7598705
3: -14.0013285, -12.0967751, -13.9756250, -12.1190510, -1.5845590, 1.5825992
4: -5.5931997, -3.6559031, -5.6169357, -3.7175634, -1.8756363, 1.9610326
5: -7.0611372, -5.5880609, -7.0511684, -5.5979013, -1.2764080, 1.3129859
6: 8.2899466, 10.0849972, 8.2757215, 10.0315580, -1.5571785, 1.6602082
7: -14.0177317, -12.1141577, -14.0067368, -12.1321783, -1.4803076, 1.4769933
8: -6.0839720, -4.5808887, -6.0984664, -4.6512837, -1.1129458, 1.1703949
9: -10.8831568, -8.5435457, -10.8352194, -8.5111790, -2.1709466, 2.1252837

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0556912, upper bound: 1.0555301
time: 6.97 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0556912, upper bound: 1.0622463
time: 3.80 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6440289, 0.1170566, -1.6454377, 0.1223217, -1.6748428, 1.6713800
1: -17.9686394, -15.6648798, -17.9519958, -15.6329288, -1.9504662, 1.8990045
2: -6.5292134, -4.4837713, -6.5841846, -4.5023756, -1.7504783, 1.8121290
3: -13.9683418, -12.1419897, -14.0102587, -12.1190176, -1.5557184, 1.5742130
4: -5.5857277, -3.7300534, -5.5949364, -3.6901467, -1.8955810, 1.8648829
5: -7.0527544, -5.6105976, -7.0408001, -5.5895252, -1.2802823, 1.2574587
6: 8.2773838, 10.0093632, 8.3170605, 10.1020603, -1.6500731, 1.5120127
7: -14.0086727, -12.1354914, -14.0330000, -12.1235991, -1.4569173, 1.4813738
8: -6.0861721, -4.6511135, -6.0786629, -4.6322737, -1.1050063, 1.0828106
9: -10.8196745, -8.5050039, -10.8480673, -8.5354233, -2.0699892, 2.1320229

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B2_A1_B1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0740062, upper bound: 1.0512663
time: 3.86 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0740062, upper bound: 1.0512662
time: 4.14 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6464379, 0.1223224, -1.6763024, 1.6725760
1: -17.9605503, -15.6407909, -17.9519958, -15.6262808, -1.9533186, 1.9231596
2: -6.5333314, -4.4837227, -6.5850840, -4.5023246, -1.7554150, 1.8136072
3: -13.9777908, -12.1190071, -14.0102634, -12.1150637, -1.5644922, 1.5829272
4: -5.6204309, -3.7172384, -5.6012630, -3.6896396, -1.9307914, 1.8840246
5: -7.0514894, -5.5967188, -7.0412030, -5.5858030, -1.2967246, 1.2518587
6: 8.2744904, 10.0354328, 8.3170557, 10.1077023, -1.6687317, 1.5123546
7: -14.0094528, -12.1319809, -14.0331659, -12.1226940, -1.4580603, 1.4842772
8: -6.1021070, -4.6512437, -6.0823116, -4.6322727, -1.1121470, 1.0889201
9: -10.8361607, -8.5088787, -10.8520432, -8.5354223, -2.0863829, 2.1353068

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B2_A1_B1_A2_A1

### Relational analysis result of IS_B2_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0740240, upper bound: 1.0557640
time: 3.80 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2

### Relational analysis result of IS_B2_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0740240, upper bound: 1.0557639
time: 3.72 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6436982, 0.1163945, -1.6732104, 0.1238357, -1.6738000, 1.6999102
1: -17.9686394, -15.6680336, -17.9781685, -15.6414309, -1.9484611, 1.9237099
2: -6.5291390, -4.4842200, -6.5849137, -4.4929523, -1.7549367, 1.8284855
3: -13.9661732, -12.1420336, -14.0013247, -12.1007271, -1.5738273, 1.5757885
4: -5.5822315, -3.7303596, -5.5868731, -3.6564279, -1.9258037, 1.8565135
5: -7.0524511, -5.6117826, -7.0607500, -5.5917835, -1.2965522, 1.2819264
6: 8.2786198, 10.0054932, 8.2899513, 10.0793514, -1.6415472, 1.5567222
7: -14.0059566, -12.1356945, -14.0175657, -12.1150713, -1.4759016, 1.4774082
8: -6.0825377, -4.6511531, -6.0802822, -4.5808878, -1.1632545, 1.1068380
9: -10.8187342, -8.5073051, -10.8792028, -8.5435448, -2.1088829, 2.1677508

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_B2_A1_B2_A1_A1

### Relational analysis result of IS_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0410558, upper bound: 1.0138257
time: 3.81 seconds

## Relational analysis of IS_B2_A1_B2_A1_A2

### Relational analysis result of IS_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0487069, upper bound: 1.0366572
time: 3.77 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6447271, 0.1117580, -1.6742057, 0.1238362, -1.6752715, 1.7010980
1: -17.9605484, -15.6439562, -17.9781647, -15.6348305, -1.9514585, 1.9478774
2: -6.5332546, -4.4841700, -6.5858092, -4.4928999, -1.7598705, 1.8299561
3: -13.9756250, -12.1190510, -14.0013285, -12.0967751, -1.5825992, 1.5845590
4: -5.6169357, -3.7175634, -5.5931997, -3.6559031, -1.9610326, 1.8756363
5: -7.0511684, -5.5979013, -7.0611372, -5.5880609, -1.3129861, 1.2764080
6: 8.2757215, 10.0315580, 8.2899466, 10.0849972, -1.6602087, 1.5571785
7: -14.0067368, -12.1321783, -14.0177317, -12.1141577, -1.4769931, 1.4803076
8: -6.0984664, -4.6512837, -6.0839720, -4.5808887, -1.1703949, 1.1129457
9: -10.8352194, -8.5111790, -10.8831568, -8.5435457, -2.1252832, 2.1709466

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0555301, upper bound: 1.0556913
time: 4.20 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0555301, upper bound: 1.0557638
time: 4.40 seconds

## BFS IS instance: IS_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6454377, 0.1223217, -1.6804867, 1.6941271
1: -17.9600925, -15.6518440, -17.9519958, -15.6329288, -1.9711132, 1.9411051
2: -6.5795999, -4.5005484, -6.5841846, -4.5023756, -1.7554426, 1.7623825
3: -14.0056801, -12.1403723, -14.0102587, -12.1190176, -1.5887437, 1.5727897
4: -5.5723267, -3.7033584, -5.5949364, -3.6901467, -1.8821800, 1.8915780
5: -7.0431242, -5.6016593, -7.0408001, -5.5895252, -1.3141766, 1.3071386
6: 8.3133726, 10.0787506, 8.3170605, 10.1020603, -1.5600595, 1.5458212
7: -14.0346870, -12.1278963, -14.0330000, -12.1235991, -1.4812834, 1.4721909
8: -6.0788412, -4.6320820, -6.0786629, -4.6322737, -1.0896621, 1.0878537
9: -10.8304892, -8.5228643, -10.8480673, -8.5354233, -2.0980148, 2.1309781

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B2_A2_B1_A1_A1

### Relational analysis result of IS_B2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0675086, upper bound: 1.0454817
time: 5.27 seconds

## Relational analysis of IS_B2_A2_B1_A1_A2

### Relational analysis result of IS_B2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0675086, upper bound: 1.0454813
time: 4.45 seconds

## BFS IS instance: IS_B2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6464379, 0.1223224, -1.6820087, 1.6953263
1: -17.9519997, -15.6275787, -17.9519958, -15.6262808, -1.9747105, 1.9629755
2: -6.5837455, -4.5005035, -6.5850840, -4.5023246, -1.7602863, 1.7638578
3: -14.0151215, -12.1173935, -14.0102634, -12.1150637, -1.5965400, 1.5827284
4: -5.6089945, -3.6892047, -5.6012630, -3.6896396, -1.9193549, 1.9120584
5: -7.0418663, -5.5878100, -7.0412030, -5.5858030, -1.3322177, 1.2998352
6: 8.3123579, 10.1027765, 8.3170557, 10.1077023, -1.5792592, 1.5441871
7: -14.0354652, -12.1244431, -14.0331659, -12.1226940, -1.4824228, 1.4752610
8: -6.0946827, -4.6322107, -6.0823116, -4.6322727, -1.0968930, 1.0939749
9: -10.8474274, -8.5267429, -10.8520432, -8.5354223, -2.1144066, 2.1340928

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B2_A2_B1_A2_A1

### Relational analysis result of IS_B2_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0675624, upper bound: 1.0512296
time: 3.63 seconds

## Relational analysis of IS_B2_A2_B1_A2_A2

### Relational analysis result of IS_B2_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0675624, upper bound: 1.0512296
time: 3.81 seconds

## BFS IS instance: IS_B2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6406817, 0.1302183, -1.6732104, 0.1238357, -1.6794443, 1.7226591
1: -17.9600906, -15.6549997, -17.9781685, -15.6414309, -1.9696302, 1.9659247
2: -6.5795259, -4.5009937, -6.5849137, -4.4929523, -1.7597661, 1.7787418
3: -14.0035057, -12.1404171, -14.0013247, -12.1007271, -1.6070910, 1.5743139
4: -5.5688229, -3.7036729, -5.5868731, -3.6564279, -1.9123950, 1.8832002
5: -7.0428200, -5.6028433, -7.0607500, -5.5917835, -1.3301830, 1.3314595
6: 8.3145847, 10.0748777, 8.2899513, 10.0793514, -1.5514698, 1.5901291
7: -14.0319691, -12.1280909, -14.0175657, -12.1150713, -1.5002673, 1.4682405
8: -6.0752068, -4.6321192, -6.0802822, -4.5808878, -1.1478465, 1.1115923
9: -10.8295345, -8.5251560, -10.8792028, -8.5435448, -2.1369014, 2.1669631

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_B2_A2_B2_A1_A1

### Relational analysis result of IS_B2_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0358414, upper bound: 1.0081872
time: 4.01 seconds

## Relational analysis of IS_B2_A2_B2_A1_A2

### Relational analysis result of IS_B2_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0410820, upper bound: 1.0313425
time: 3.75 seconds

## BFS IS instance: IS_B2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6416143, 0.1255802, -1.6742057, 0.1238362, -1.6809845, 1.7238493
1: -17.9519958, -15.6307487, -17.9781647, -15.6348305, -1.9733725, 1.9878008
2: -6.5836663, -4.5009475, -6.5858092, -4.4928999, -1.7646070, 1.7802072
3: -14.0129499, -12.1174355, -14.0013285, -12.0967751, -1.6148853, 1.5843072
4: -5.6054897, -3.6895304, -5.5931997, -3.6559031, -1.9495866, 1.9036694
5: -7.0415478, -5.5889931, -7.0611372, -5.5880609, -1.3482151, 1.3242373
6: 8.3135643, 10.0988979, 8.2899466, 10.0849972, -1.5706725, 1.5886061
7: -14.0327482, -12.1246357, -14.0177317, -12.1141577, -1.5013547, 1.4713056
8: -6.0910435, -4.6322494, -6.0839720, -4.5808887, -1.1550720, 1.1177118
9: -10.8464746, -8.5290337, -10.8831568, -8.5435457, -2.1532993, 2.1699901

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B2_A2_B2_A2_B1

### Relational analysis result of IS_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0511604
time: 4.04 seconds

## Relational analysis of IS_B2_A2_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0512298
time: 5.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 15.66 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0819898
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0819898
IS_B1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0703562, upper bound: 1.0820436
IS_B1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0703562, upper bound: 1.0820438
IS_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0266211, upper bound: 1.0491982
IS_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0497349, upper bound: 1.0567052
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0703074, upper bound: 1.0633820
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0703074, upper bound: 1.0703562
IS_B1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0512660, upper bound: 1.0740079
IS_B1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0512660, upper bound: 1.0740079
IS_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0557637, upper bound: 1.0740257
IS_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0557637, upper bound: 1.0740235
IS_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0138255, upper bound: 1.0410554
IS_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0366568, upper bound: 1.0487070
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0556912, upper bound: 1.0555301
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0556912, upper bound: 1.0622463
IS_B2_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0740062, upper bound: 1.0512663
IS_B2_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0740062, upper bound: 1.0512662
IS_B2_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0740240, upper bound: 1.0557640
IS_B2_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0740240, upper bound: 1.0557639
IS_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0410558, upper bound: 1.0138257
IS_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0487069, upper bound: 1.0366572
IS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0555301, upper bound: 1.0556913
IS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0555301, upper bound: 1.0557638
IS_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0675086, upper bound: 1.0454817
IS_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0675086, upper bound: 1.0454813
IS_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0675624, upper bound: 1.0512296
IS_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0675624, upper bound: 1.0512296
IS_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0358414, upper bound: 1.0081872
IS_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0410820, upper bound: 1.0313425
IS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0511604
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 15.66
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0512298

## BFS IS instance: IS_B1_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -1.6485716, 0.1085088, -1.6429667, 0.1131482, -1.6667690, 1.6562080
1: -17.9605465, -15.6460991, -17.9686394, -15.6773357, -1.8909364, 1.9330871
2: -6.5337744, -4.4856172, -6.5289249, -4.4858875, -1.7349367, 1.7298708
3: -13.9729290, -12.1206341, -13.9634705, -12.1421843, -1.5331354, 1.5430608
4: -5.6072845, -3.7181706, -5.5738678, -3.7312579, -1.8760266, 1.8556972
5: -7.0504222, -5.5984144, -7.0514927, -5.6163478, -1.2558331, 1.2698219
6: 8.2793627, 10.0337639, 8.2822876, 10.0031853, -1.5379870, 1.5584230
7: -14.0069876, -12.1311550, -14.0060873, -12.1362534, -1.4542875, 1.4582250
8: -6.0860777, -4.6513085, -6.0666018, -4.6511798, -1.1004370, 1.0812157
9: -10.8368025, -8.5175915, -10.8155832, -8.5137215, -2.1126041, 2.0850539

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_B1_A1_A1_B1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0421525, upper bound: 1.0470989
time: 4.09 seconds

## Relational analysis of IS_B1_A1_A1_B1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0497348, upper bound: 1.0683625
time: 3.93 seconds

## BFS IS instance: IS_B1_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -1.6485716, 0.1085088, -1.6710336, 0.1146619, -1.6663432, 1.6841822
1: -17.9605465, -15.6460991, -17.9948082, -15.6855450, -1.8863611, 1.9614248
2: -6.5337744, -4.4856172, -6.5296888, -4.4764652, -1.7417002, 1.7310538
3: -13.9729290, -12.1206341, -13.9545717, -12.1238918, -1.5530090, 1.5373187
4: -5.6072845, -3.7181706, -5.5659685, -3.6974487, -1.9098358, 1.8477979
5: -7.0504222, -5.5984144, -7.0716925, -5.6187625, -1.2585397, 1.2951777
6: 8.2793627, 10.0337639, 8.2547817, 9.9804878, -1.5431719, 1.6050406
7: -14.0069876, -12.1311550, -13.9906483, -12.1276608, -1.4741614, 1.4567199
8: -6.0860777, -4.6513085, -6.0681472, -4.5998230, -1.1618061, 1.0887649
9: -10.8368025, -8.5175915, -10.8470211, -8.5219250, -2.1049733, 2.1253829

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_B1_A1_A1_B1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0421525, upper bound: 1.0470971
time: 3.85 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0497348, upper bound: 1.0683624
time: 4.01 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.6495955, 0.1085103, -1.6439968, 0.1085086, -1.6679502, 1.6576524
1: -17.9605465, -15.6395464, -17.9605465, -15.6532793, -1.9137087, 1.9359150
2: -6.5346708, -4.4855671, -6.5330286, -4.4858408, -1.7364559, 1.7347794
3: -13.9729347, -12.1166830, -13.9729214, -12.1192026, -1.5422330, 1.5518284
4: -5.6134143, -3.7176580, -5.6085730, -3.7185049, -1.8949094, 1.8909149
5: -7.0508256, -5.5946903, -7.0501585, -5.6024623, -1.2502382, 1.2863994
6: 8.2793570, 10.0396023, 8.2793722, 10.0292358, -1.5365882, 1.5772882
7: -14.0071535, -12.1302347, -14.0068674, -12.1327229, -1.4571896, 1.4591200
8: -6.0897551, -4.6513085, -6.0825377, -4.6513100, -1.1065433, 1.0885115
9: -10.8406801, -8.5175896, -10.8320770, -8.5175905, -2.1157155, 2.1014118

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B1_A1_A1_B2_B1_A1

### Relational analysis result of IS_B1_A1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0750604
time: 4.09 seconds

## Relational analysis of IS_B1_A1_A1_B2_B1_A2

### Relational analysis result of IS_B1_A1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0820433
time: 4.34 seconds

## BFS IS instance: IS_B1_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.6495955, 0.1085103, -1.6720339, 0.1100070, -1.6674895, 1.6856685
1: -17.9605465, -15.6395464, -17.9867172, -15.6616745, -1.9091797, 1.9642487
2: -6.5346708, -4.4855671, -6.5337782, -4.4764175, -1.7432179, 1.7359018
3: -13.9729347, -12.1166830, -13.9640217, -12.1008959, -1.5621066, 1.5460382
4: -5.6134143, -3.7176580, -5.6006718, -3.6847847, -1.9286296, 1.8830137
5: -7.0508256, -5.5946903, -7.0701571, -5.6048150, -1.2529454, 1.3116536
6: 8.2793570, 10.0396023, 8.2517681, 10.0065508, -1.5418344, 1.6238070
7: -14.0071535, -12.1302347, -13.9914274, -12.1240950, -1.4769835, 1.4576154
8: -6.0897551, -4.6513085, -6.0840840, -4.5999527, -1.1678975, 1.0960063
9: -10.8406801, -8.5175896, -10.8633537, -8.5257845, -2.1080666, 2.1412725

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B1_A1_A1_B2_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0750607
time: 4.08 seconds

## Relational analysis of IS_B1_A1_A1_B2_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0820437
time: 4.53 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -1.6755145, 0.1047156, -1.6391675, 0.0843215, -1.6720424, 1.6548133
1: -17.9708710, -15.6590939, -17.9011307, -15.6538582, -1.8035483, 1.8106430
2: -6.5324259, -4.4945116, -6.5208983, -4.5897617, -1.6451333, 1.7022026
3: -13.9532337, -12.1024189, -13.9058933, -12.1585732, -1.5108643, 1.5082769
4: -5.5896807, -3.6872830, -5.5276499, -3.7290661, -1.8606145, 1.8403668
5: -7.0638695, -5.6008182, -7.0160065, -5.5973926, -1.2838473, 1.2355907
6: 8.2572002, 10.0089340, 8.3108559, 10.0071163, -1.5762877, 1.4958286
7: -13.9915218, -12.1459122, -14.0000019, -12.2630014, -1.2955198, 1.4147990
8: -6.0786381, -4.5999784, -6.0329714, -4.6372843, -1.1298419, 1.1094220
9: -10.8549576, -8.5261984, -10.7529211, -8.5099058, -2.1313720, 2.0467777

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B1_A1_A2_B1_B1_B1

### Relational analysis result of IS_B1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0266211, upper bound: 1.0491982
time: 4.00 seconds

## Relational analysis of IS_B1_A1_A2_B1_B1_B2

### Relational analysis result of IS_B1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0266211, upper bound: 1.0491982
time: 3.91 seconds

## BFS IS instance: IS_B1_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -1.6766319, 0.1100085, -1.6426257, 0.1099414, -1.6912475, 1.6555600
1: -17.9867172, -15.6544209, -17.9398155, -15.6726179, -1.8971009, 1.8155408
2: -6.5345006, -4.4761944, -6.5274253, -4.5050497, -1.6776948, 1.7375712
3: -13.9640303, -12.1023426, -13.9616137, -12.1423378, -1.5344248, 1.5180862
4: -5.5993857, -3.6844392, -5.5634737, -3.7331753, -1.8662105, 1.8790345
5: -7.0704103, -5.6007915, -7.0379629, -5.6120901, -1.2891085, 1.2582967
6: 8.2517605, 10.0110798, 8.3040924, 10.0036402, -1.5942883, 1.5235069
7: -13.9915466, -12.1225433, -14.0055456, -12.1558323, -1.2901325, 1.4833312
8: -6.0875735, -4.5999537, -6.0624261, -4.6515675, -1.1239351, 1.1415284
9: -10.8681459, -8.5257854, -10.8038292, -8.5078650, -2.1695805, 2.1058254

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: B, layer: 3, pos: 2461

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 577

## Relational analysis of IS_B1_A1_A2_B1_B2_B1

### Relational analysis result of IS_B1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.9979732, upper bound: 1.0271819
time: 3.61 seconds

## Relational analysis of IS_B1_A1_A2_B1_B2_B2

### Relational analysis result of IS_B1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0463969, upper bound: 1.0533441
time: 4.09 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6710336, 0.1146619, -1.6447271, 0.1117580, -1.6895003, 1.6661220
1: -17.9948082, -15.6855450, -17.9605484, -15.6439562, -1.9663305, 1.8892303
2: -6.5296888, -4.4764652, -6.5332546, -4.4841700, -1.7528791, 1.7440033
3: -13.9545717, -12.1238918, -13.9756250, -12.1190510, -1.5453033, 1.5572395
4: -5.5659685, -3.6974487, -5.6169357, -3.7175634, -1.8484051, 1.9194870
5: -7.0716925, -5.6187625, -7.0511684, -5.5979013, -1.3074508, 1.2735028
6: 8.2547817, 9.9804878, 8.2757215, 10.0315580, -1.6228819, 1.5385981
7: -13.9906483, -12.1276608, -14.0067368, -12.1321783, -1.4538546, 1.4802091
8: -6.0681472, -4.5998230, -6.0984664, -4.6512837, -1.1052444, 1.1776378
9: -10.8470211, -8.5219250, -10.8352194, -8.5111790, -2.1387553, 2.1618500

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0266211, upper bound: 1.0421525
time: 3.92 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0497349, upper bound: 1.0497351
time: 4.23 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6720339, 0.1100070, -1.6447271, 0.1117580, -1.6909823, 1.6580315
1: -17.9867172, -15.6616745, -17.9605484, -15.6439562, -1.9499192, 1.9120259
2: -6.5337782, -4.4764175, -6.5332546, -4.4841700, -1.7581758, 1.7444425
3: -13.9640217, -12.1008959, -13.9756250, -12.1190510, -1.5439544, 1.5663338
4: -5.6006718, -3.6847847, -5.6169357, -3.7175634, -1.8831084, 1.9321511
5: -7.0701571, -5.6048150, -7.0511684, -5.5979013, -1.2820694, 1.2680280
6: 8.2517681, 10.0065508, 8.2757215, 10.0315580, -1.5966191, 1.5372174
7: -13.9914274, -12.1240950, -14.0067368, -12.1321783, -1.4540200, 1.4817801
8: -6.0840840, -4.5999527, -6.0984664, -4.6512837, -1.1125909, 1.1677897
9: -10.8633537, -8.5257845, -10.8352194, -8.5111790, -2.1546392, 2.1520534

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0266211, upper bound: 1.0491983
time: 3.95 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0497349, upper bound: 1.0568088
time: 4.56 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B1

### Backsubstitution after applying IS history:
0: -1.6454377, 0.1223217, -1.6429667, 0.1131482, -1.6669121, 1.6734037
1: -17.9519958, -15.6329288, -17.9686394, -15.6773357, -1.8874669, 1.9504604
2: -6.5841846, -4.5023756, -6.5289249, -4.4858875, -1.8052874, 1.7455235
3: -14.0102587, -12.1190176, -13.9634705, -12.1421843, -1.5739889, 1.5497670
4: -5.5949364, -3.6901467, -5.5738678, -3.7312579, -1.8636785, 1.8837211
5: -7.0408001, -5.5895252, -7.0514927, -5.6163478, -1.2482080, 1.2764733
6: 8.3170605, 10.1020603, 8.2822876, 10.0031853, -1.4969609, 1.6409259
7: -14.0330000, -12.1235991, -14.0060873, -12.1362534, -1.4794073, 1.4497912
8: -6.0786629, -4.6322737, -6.0666018, -4.6511798, -1.0827245, 1.0838687
9: -10.8480673, -8.5354233, -10.8155832, -8.5137215, -2.1149139, 2.0582805

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_B1_A2_A1_B1_B1_B1

### Relational analysis result of IS_B1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0138255, upper bound: 1.0543786
time: 3.90 seconds

## Relational analysis of IS_B1_A2_A1_B1_B1_B2

### Relational analysis result of IS_B1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0366568, upper bound: 1.0607087
time: 5.95 seconds

## BFS IS instance: IS_B1_A2_A1_B1_B2

### Backsubstitution after applying IS history:
0: -1.6454377, 0.1223217, -1.6710336, 0.1146619, -1.6664863, 1.7013779
1: -17.9519958, -15.6329288, -17.9948082, -15.6855450, -1.8828917, 1.9787984
2: -6.5841846, -4.5023756, -6.5296888, -4.4764652, -1.8120508, 1.7467060
3: -14.0102587, -12.1190176, -13.9545717, -12.1238918, -1.5938625, 1.5440245
4: -5.5949364, -3.6901467, -5.5659685, -3.6974487, -1.8974876, 1.8758218
5: -7.0408001, -5.5895252, -7.0716925, -5.6187625, -1.2509146, 1.3018291
6: 8.3170605, 10.1020603, 8.2547817, 9.9804878, -1.5021458, 1.6875434
7: -14.0330000, -12.1235991, -13.9906483, -12.1276608, -1.4992812, 1.4482861
8: -6.0786629, -4.6322737, -6.0681472, -4.5998230, -1.1440935, 1.0914179
9: -10.8480673, -8.5354233, -10.8470211, -8.5219250, -2.1072822, 2.0986094

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_B1_A2_A1_B1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0313498, upper bound: 1.0392504
time: 5.53 seconds

## Relational analysis of IS_B1_A2_A1_B1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0366567, upper bound: 1.0607091
time: 6.15 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.6464379, 0.1223224, -1.6439968, 0.1085086, -1.6681046, 1.6748462
1: -17.9519958, -15.6262808, -17.9605465, -15.6532793, -1.9116387, 1.9533124
2: -6.5850840, -4.5023246, -6.5330286, -4.4858408, -1.8067651, 1.7504311
3: -14.0102634, -12.1150637, -13.9729214, -12.1192026, -1.5827041, 1.5585351
4: -5.6012630, -3.6896396, -5.6085730, -3.7185049, -1.8827581, 1.9189334
5: -7.0412030, -5.5858030, -7.0501585, -5.6024623, -1.2426074, 1.2928874
6: 8.3170557, 10.1077023, 8.2793722, 10.0292358, -1.4972591, 1.6595860
7: -14.0331659, -12.1226940, -14.0068674, -12.1327229, -1.4823041, 1.4509351
8: -6.0823116, -4.6322727, -6.0825377, -4.6513100, -1.0888324, 1.0910716
9: -10.8520432, -8.5354223, -10.8320770, -8.5175905, -2.1181893, 2.0746384

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B1_A2_A1_B2_B1_A1

### Relational analysis result of IS_B1_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0673021
time: 4.24 seconds

## Relational analysis of IS_B1_A2_A1_B2_B1_A2

### Relational analysis result of IS_B1_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0740240
time: 4.27 seconds

## BFS IS instance: IS_B1_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.6464379, 0.1223224, -1.6720339, 0.1100070, -1.6676440, 1.7028623
1: -17.9519958, -15.6262808, -17.9867172, -15.6616745, -1.9071097, 1.9816461
2: -6.5850840, -4.5023246, -6.5337782, -4.4764175, -1.8135276, 1.7515540
3: -14.0102634, -12.1150637, -13.9640217, -12.1008959, -1.6025777, 1.5527444
4: -5.6012630, -3.6896396, -5.6006718, -3.6847847, -1.9164784, 1.9110322
5: -7.0412030, -5.5858030, -7.0701571, -5.6048150, -1.2453146, 1.3181417
6: 8.3170557, 10.1077023, 8.2517681, 10.0065508, -1.5025048, 1.7061048
7: -14.0331659, -12.1226940, -13.9914274, -12.1240950, -1.5020976, 1.4494302
8: -6.0823116, -4.6322727, -6.0840840, -4.5999527, -1.1501865, 1.0985665
9: -10.8520432, -8.5354223, -10.8633537, -8.5257845, -2.1105404, 2.1144996

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B1_A2_A1_B2_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0673003
time: 4.83 seconds

## Relational analysis of IS_B1_A2_A1_B2_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0673002
time: 6.14 seconds

## BFS IS instance: IS_B1_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -1.6721765, 0.1185374, -1.6391675, 0.0843215, -1.6717825, 1.6720910
1: -17.9623203, -15.6461792, -17.9011307, -15.6538582, -1.8000722, 1.8273573
2: -6.5828109, -4.5108356, -6.5208983, -4.5897617, -1.7154057, 1.7181368
3: -13.9905815, -12.1007996, -13.9058933, -12.1585732, -1.5520148, 1.5149345
4: -5.5777483, -3.6592672, -5.5276499, -3.7290661, -1.8486822, 1.8683827
5: -7.0542655, -5.5918217, -7.0160065, -5.5973926, -1.2765241, 1.2425392
6: 8.2950048, 10.0771313, 8.3108559, 10.0071163, -1.5353894, 1.5782642
7: -14.0175419, -12.1379604, -14.0000019, -12.2630014, -1.3206558, 1.4068477
8: -6.0714030, -4.5809131, -6.0329714, -4.6372843, -1.1121019, 1.1121264
9: -10.8655968, -8.5439348, -10.7529211, -8.5099058, -2.1326318, 2.0200667

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B1_A2_A2_B1_B1_B1

### Relational analysis result of IS_B1_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0138255, upper bound: 1.0410553
time: 4.43 seconds

## Relational analysis of IS_B1_A2_A2_B1_B1_B2

### Relational analysis result of IS_B1_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0138255, upper bound: 1.0410554
time: 4.58 seconds

## BFS IS instance: IS_B1_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -1.6732104, 0.1238357, -1.6426257, 0.1099414, -1.6910462, 1.6728053
1: -17.9781685, -15.6414309, -17.9398155, -15.6726179, -1.8936396, 1.8335669
2: -6.5849137, -4.4929523, -6.5274253, -4.5050497, -1.7494507, 1.7531762
3: -14.0013247, -12.1007271, -13.9616137, -12.1423378, -1.5753927, 1.5248852
4: -5.5868731, -3.6564279, -5.5634737, -3.7331753, -1.8536978, 1.9070458
5: -7.0607500, -5.5917835, -7.0379629, -5.6120901, -1.2815802, 1.2654140
6: 8.2899513, 10.0793514, 8.3040924, 10.0036402, -1.5531464, 1.6068745
7: -14.0175657, -12.1150713, -14.0055456, -12.1558323, -1.3141642, 1.4752519
8: -6.0802822, -4.5808878, -6.0624261, -4.6515675, -1.1062399, 1.1449345
9: -10.8792028, -8.5435448, -10.8038292, -8.5078650, -2.1723633, 2.0790548

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: B, layer: 3, pos: 2461

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 577

## Relational analysis of IS_B1_A2_A2_B1_B2_B1

### Relational analysis result of IS_B1_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.9843721, upper bound: 1.0185311
time: 3.75 seconds

## Relational analysis of IS_B1_A2_A2_B1_B2_B2

### Relational analysis result of IS_B1_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0332768, upper bound: 1.0453862
time: 4.27 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6677475, 0.1284892, -1.6447271, 0.1117580, -1.6892152, 1.6833496
1: -17.9862614, -15.6726294, -17.9605484, -15.6439562, -1.9642696, 1.9071293
2: -6.5800805, -4.4932203, -6.5332546, -4.4841700, -1.8231983, 1.7596097
3: -13.9918699, -12.1222744, -13.9756250, -12.1190510, -1.5858784, 1.5638967
4: -5.5523729, -3.6708283, -5.6169357, -3.7175634, -1.8348095, 1.9461074
5: -7.0620794, -5.6097002, -7.0511684, -5.5979013, -1.2999673, 1.2811785
6: 8.2910995, 10.0498381, 8.2757215, 10.0315580, -1.5834460, 1.6222105
7: -14.0166693, -12.1201248, -14.0067368, -12.1321783, -1.4790230, 1.4722102
8: -6.0609593, -4.5807586, -6.0984664, -4.6512837, -1.0875888, 1.1802263
9: -10.8575659, -8.5396843, -10.8352194, -8.5111790, -2.1398063, 2.1350794

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0136093, upper bound: 1.0343456
time: 4.47 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0364477, upper bound: 1.0419815
time: 4.41 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6686337, 0.1238341, -1.6447271, 0.1117580, -1.6907759, 1.6752563
1: -17.9781666, -15.6485596, -17.9605484, -15.6439562, -1.9478579, 1.9299779
2: -6.5841751, -4.4931755, -6.5332546, -4.4841700, -1.8283806, 1.7600470
3: -14.0013113, -12.0992775, -13.9756250, -12.1190510, -1.5845480, 1.5729899
4: -5.5890379, -3.6567700, -5.6169357, -3.7175634, -1.8714745, 1.9601657
5: -7.0604973, -5.5957823, -7.0511684, -5.5979013, -1.2745461, 1.2757032
6: 8.2899628, 10.0738573, 8.2757215, 10.0315580, -1.5571737, 1.6207557
7: -14.0174465, -12.1166172, -14.0067368, -12.1321783, -1.4791832, 1.4740236
8: -6.0768023, -4.5808878, -6.0984664, -4.6512837, -1.0948828, 1.1703784
9: -10.8743534, -8.5435457, -10.8352194, -8.5111790, -2.1564932, 2.1252823

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0136093, upper bound: 1.0410618
time: 4.08 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0364477, upper bound: 1.0487158
time: 3.69 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A1

### Backsubstitution after applying IS history:
0: -1.6429667, 0.1131482, -1.6454377, 0.1223217, -1.6734037, 1.6669121
1: -17.9686394, -15.6773357, -17.9519958, -15.6329288, -1.9504604, 1.8874669
2: -6.5289249, -4.4858875, -6.5841846, -4.5023756, -1.7455235, 1.8052874
3: -13.9634705, -12.1421843, -14.0102587, -12.1190176, -1.5497665, 1.5739889
4: -5.5738678, -3.7312579, -5.5949364, -3.6901467, -1.8837211, 1.8636785
5: -7.0514927, -5.6163478, -7.0408001, -5.5895252, -1.2764733, 1.2482080
6: 8.2822876, 10.0031853, 8.3170605, 10.1020603, -1.6409259, 1.4969611
7: -14.0060873, -12.1362534, -14.0330000, -12.1235991, -1.4497910, 1.4794071
8: -6.0666018, -4.6511798, -6.0786629, -4.6322737, -1.0838686, 1.0827245
9: -10.8155832, -8.5137215, -10.8480673, -8.5354233, -2.0582809, 2.1149135

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_B2_A1_B1_A1_A1_A1

### Relational analysis result of IS_B2_A1_B1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0543769, upper bound: 1.0138255
time: 3.91 seconds

## Relational analysis of IS_B2_A1_B1_A1_A1_A2

### Relational analysis result of IS_B2_A1_B1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0607089, upper bound: 1.0366569
time: 3.86 seconds

## BFS IS instance: IS_B2_A1_B1_A1_A2

### Backsubstitution after applying IS history:
0: -1.6710336, 0.1146619, -1.6454377, 0.1223217, -1.7013779, 1.6664863
1: -17.9948082, -15.6855450, -17.9519958, -15.6329288, -1.9787984, 1.8828917
2: -6.5296888, -4.4764652, -6.5841846, -4.5023756, -1.7467060, 1.8120508
3: -13.9545717, -12.1238918, -14.0102587, -12.1190176, -1.5440245, 1.5938625
4: -5.5659685, -3.6974487, -5.5949364, -3.6901467, -1.8758218, 1.8974876
5: -7.0716925, -5.6187625, -7.0408001, -5.5895252, -1.3018289, 1.2509146
6: 8.2547817, 9.9804878, 8.3170605, 10.1020603, -1.6875434, 1.5021460
7: -13.9906483, -12.1276608, -14.0330000, -12.1235991, -1.4482861, 1.4992814
8: -6.0681472, -4.5998230, -6.0786629, -4.6322737, -1.0914179, 1.1440935
9: -10.8470211, -8.5219250, -10.8480673, -8.5354233, -2.0986099, 2.1072822

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 2480

## Relational analysis of IS_B2_A1_B1_A1_A2_B1

### Relational analysis result of IS_B2_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0392506, upper bound: 1.0313500
time: 4.02 seconds

## Relational analysis of IS_B2_A1_B1_A1_A2_B2

### Relational analysis result of IS_B2_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0607091, upper bound: 1.0366569
time: 3.68 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -1.6439968, 0.1085086, -1.6464379, 0.1223224, -1.6748462, 1.6681046
1: -17.9605465, -15.6532793, -17.9519958, -15.6262808, -1.9533124, 1.9116390
2: -6.5330286, -4.4858408, -6.5850840, -4.5023246, -1.7504311, 1.8067651
3: -13.9729214, -12.1192026, -14.0102634, -12.1150637, -1.5585351, 1.5827041
4: -5.6085730, -3.7185049, -5.6012630, -3.6896396, -1.9189334, 1.8827581
5: -7.0501585, -5.6024623, -7.0412030, -5.5858030, -1.2928874, 1.2426074
6: 8.2793722, 10.0292358, 8.3170557, 10.1077023, -1.6595860, 1.4972589
7: -14.0068674, -12.1327229, -14.0331659, -12.1226940, -1.4509349, 1.4823039
8: -6.0825377, -4.6513100, -6.0823116, -4.6322727, -1.0910717, 1.0888324
9: -10.8320770, -8.5175905, -10.8520432, -8.5354223, -2.0746384, 2.1181898

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B2_A1_B1_A2_A1_B1

### Relational analysis result of IS_B2_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0673004, upper bound: 1.0556912
time: 3.96 seconds

## Relational analysis of IS_B2_A1_B1_A2_A1_B2

### Relational analysis result of IS_B2_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0673004, upper bound: 1.0557639
time: 3.94 seconds

## BFS IS instance: IS_B2_A1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -1.6720339, 0.1100070, -1.6464379, 0.1223224, -1.7028623, 1.6676440
1: -17.9867172, -15.6616745, -17.9519958, -15.6262808, -1.9816461, 1.9071097
2: -6.5337782, -4.4764175, -6.5850840, -4.5023246, -1.7515540, 1.8135276
3: -13.9640217, -12.1008959, -14.0102634, -12.1150637, -1.5527449, 1.6025777
4: -5.6006718, -3.6847847, -5.6012630, -3.6896396, -1.9110322, 1.9164784
5: -7.0701571, -5.6048150, -7.0412030, -5.5858030, -1.3181417, 1.2453144
6: 8.2517681, 10.0065508, 8.3170557, 10.1077023, -1.7061048, 1.5025048
7: -13.9914274, -12.1240950, -14.0331659, -12.1226940, -1.4494305, 1.5020974
8: -6.0840840, -4.5999527, -6.0823116, -4.6322727, -1.0985667, 1.1501864
9: -10.8633537, -8.5257845, -10.8520432, -8.5354223, -2.1144991, 2.1105409

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B2_A1_B1_A2_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0673004, upper bound: 1.0556913
time: 3.98 seconds

## Relational analysis of IS_B2_A1_B1_A2_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0673004, upper bound: 1.0557640
time: 3.88 seconds

## BFS IS instance: IS_B2_A1_B2_A1_A1

### Backsubstitution after applying IS history:
0: -1.6391675, 0.0843215, -1.6721765, 0.1185374, -1.6720905, 1.6717825
1: -17.9011307, -15.6538582, -17.9623203, -15.6461792, -1.8273573, 1.8000720
2: -6.5208983, -4.5897617, -6.5828109, -4.5108356, -1.7181368, 1.7154055
3: -13.9058933, -12.1585732, -13.9905815, -12.1007996, -1.5149345, 1.5520151
4: -5.5276499, -3.7290661, -5.5777483, -3.6592672, -1.8683827, 1.8486822
5: -7.0160065, -5.5973926, -7.0542655, -5.5918217, -1.2425392, 1.2765238
6: 8.3108559, 10.0071163, 8.2950048, 10.0771313, -1.5782642, 1.5353894
7: -14.0000019, -12.2630014, -14.0175419, -12.1379604, -1.4068480, 1.3206561
8: -6.0329714, -4.6372843, -6.0714030, -4.5809131, -1.1121264, 1.1121019
9: -10.7529211, -8.5099058, -10.8655968, -8.5439348, -2.0200667, 2.1326318

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B2_A1_B2_A1_A1_A1

### Relational analysis result of IS_B2_A1_B2_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0410558, upper bound: 1.0138255
time: 3.93 seconds

## Relational analysis of IS_B2_A1_B2_A1_A1_A2

### Relational analysis result of IS_B2_A1_B2_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0410558, upper bound: 1.0138256
time: 3.89 seconds

## BFS IS instance: IS_B2_A1_B2_A1_A2

### Backsubstitution after applying IS history:
0: -1.6426257, 0.1099414, -1.6732104, 0.1238357, -1.6728053, 1.6910462
1: -17.9398155, -15.6726179, -17.9781685, -15.6414309, -1.8335669, 1.8936396
2: -6.5274253, -4.5050497, -6.5849137, -4.4929523, -1.7531762, 1.7494507
3: -13.9616137, -12.1423378, -14.0013247, -12.1007271, -1.5248852, 1.5753927
4: -5.5634737, -3.7331753, -5.5868731, -3.6564279, -1.9070458, 1.8536978
5: -7.0379629, -5.6120901, -7.0607500, -5.5917835, -1.2654142, 1.2815804
6: 8.3040924, 10.0036402, 8.2899513, 10.0793514, -1.6068745, 1.5531464
7: -14.0055456, -12.1558323, -14.0175657, -12.1150713, -1.4752517, 1.3141642
8: -6.0624261, -4.6515675, -6.0802822, -4.5808878, -1.1449347, 1.1062399
9: -10.8038292, -8.5078650, -10.8792028, -8.5435448, -2.0790548, 2.1723633

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2461

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 577

## Relational analysis of IS_B2_A1_B2_A1_A2_A1

### Relational analysis result of IS_B2_A1_B2_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0185308, upper bound: 0.9843723
time: 3.94 seconds

## Relational analysis of IS_B2_A1_B2_A1_A2_A2

### Relational analysis result of IS_B2_A1_B2_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0453844, upper bound: 1.0332771
time: 3.80 seconds

## BFS IS instance: IS_B2_A1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6447271, 0.1117580, -1.6677475, 0.1284892, -1.6833496, 1.6892152
1: -17.9605484, -15.6439562, -17.9862614, -15.6726294, -1.9071293, 1.9642696
2: -6.5332546, -4.4841700, -6.5800805, -4.4932203, -1.7596097, 1.8231983
3: -13.9756250, -12.1190510, -13.9918699, -12.1222744, -1.5638967, 1.5858784
4: -5.6169357, -3.7175634, -5.5523729, -3.6708283, -1.9461074, 1.8348095
5: -7.0511684, -5.5979013, -7.0620794, -5.6097002, -1.2811785, 1.2999673
6: 8.2757215, 10.0315580, 8.2910995, 10.0498381, -1.6222103, 1.5834460
7: -14.0067368, -12.1321783, -14.0166693, -12.1201248, -1.4722106, 1.4790232
8: -6.0984664, -4.6512837, -6.0609593, -4.5807586, -1.1802263, 1.0875888
9: -10.8352194, -8.5111790, -10.8575659, -8.5396843, -2.1350794, 2.1398063

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_B2_A1_B2_A2_B1_A1

### Relational analysis result of IS_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0343461, upper bound: 1.0181475
time: 4.40 seconds

## Relational analysis of IS_B2_A1_B2_A2_B1_A2

### Relational analysis result of IS_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -1.0419797, upper bound: 1.0410820
time: 3.72 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 13.80 seconds
IS_B1_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0421525, upper bound: 1.0470989
IS_B1_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0497348, upper bound: 1.0683625
IS_B1_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0421525, upper bound: 1.0470971
IS_B1_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0497348, upper bound: 1.0683624
IS_B1_A1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0750604
IS_B1_A1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0820433
IS_B1_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0750607
IS_B1_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0633817, upper bound: 1.0820437
IS_B1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0266211, upper bound: 1.0491982
IS_B1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0266211, upper bound: 1.0491982
IS_B1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -0.9979732, upper bound: 1.0271819
IS_B1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0463969, upper bound: 1.0533441
IS_B1_A1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0266211, upper bound: 1.0421525
IS_B1_A1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0497349, upper bound: 1.0497351
IS_B1_A1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0266211, upper bound: 1.0491983
IS_B1_A1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0497349, upper bound: 1.0568088
IS_B1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0138255, upper bound: 1.0543786
IS_B1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0366568, upper bound: 1.0607087
IS_B1_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0313498, upper bound: 1.0392504
IS_B1_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0366567, upper bound: 1.0607091
IS_B1_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0673021
IS_B1_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0740240
IS_B1_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0673003
IS_B1_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0673002
IS_B1_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0138255, upper bound: 1.0410553
IS_B1_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0138255, upper bound: 1.0410554
IS_B1_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -0.9843721, upper bound: 1.0185311
IS_B1_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0332768, upper bound: 1.0453862
IS_B1_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0136093, upper bound: 1.0343456
IS_B1_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0364477, upper bound: 1.0419815
IS_B1_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0136093, upper bound: 1.0410618
IS_B1_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0364477, upper bound: 1.0487158
IS_B2_A1_B1_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0543769, upper bound: 1.0138255
IS_B2_A1_B1_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0607089, upper bound: 1.0366569
IS_B2_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0392506, upper bound: 1.0313500
IS_B2_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0607091, upper bound: 1.0366569
IS_B2_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0673004, upper bound: 1.0556912
IS_B2_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0673004, upper bound: 1.0557639
IS_B2_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0673004, upper bound: 1.0556913
IS_B2_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0673004, upper bound: 1.0557640
IS_B2_A1_B2_A1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0410558, upper bound: 1.0138255
IS_B2_A1_B2_A1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0410558, upper bound: 1.0138256
IS_B2_A1_B2_A1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0185308, upper bound: 0.9843723
IS_B2_A1_B2_A1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0453844, upper bound: 1.0332771
IS_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0343461, upper bound: 1.0181475
IS_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 13.80
Output dim: 6, lower bound: -1.0419797, upper bound: 1.0410820
IS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.80
Output dim: 6, lower bound: -1.0555301, upper bound: 1.0557638
IS_B2_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 13.80
Output dim: 6, lower bound: -1.0675086, upper bound: 1.0454817
IS_B2_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 13.80
Output dim: 6, lower bound: -1.0675086, upper bound: 1.0454813
IS_B2_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 13.80
Output dim: 6, lower bound: -1.0675624, upper bound: 1.0512296
IS_B2_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 13.80
Output dim: 6, lower bound: -1.0675624, upper bound: 1.0512296
IS_B2_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 13.80
Output dim: 6, lower bound: -1.0358414, upper bound: 1.0081872
IS_B2_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 13.80
Output dim: 6, lower bound: -1.0410820, upper bound: 1.0313425
IS_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 13.80
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0511604
IS_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 13.80
Output dim: 6, lower bound: -1.0510568, upper bound: 1.0512298
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
type: B, layer: 3, pos: 2130
type: A, layer: 3, pos: 2130
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.31 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8153156, upper bound: 0.8125518
time: 3.96 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8153156, upper bound: 0.8153159
time: 3.90 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 8.17 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 8.17
Output dim: 6, lower bound: -0.8153156, upper bound: 0.8125518
IS_B2, status: Status.UNKNOWN, split count: 1, time: 8.17
Output dim: 6, lower bound: -0.8153156, upper bound: 0.8153159

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -1.6561922, 0.1125274, -1.6506550, 0.1124235, -1.5080442, 1.5001407
1: -17.9640560, -15.6267805, -17.9605484, -15.6270342, -1.5782843, 1.5740373
2: -6.5349898, -4.4621940, -6.5349741, -4.4834495, -1.4472051, 1.4710484
3: -13.9784403, -12.1074400, -13.9778080, -12.1164856, -1.2967229, 1.3269873
4: -5.6369295, -3.7163720, -5.6252713, -3.7163892, -1.8574867, 1.8411865
5: -7.0538297, -5.5888457, -7.0521522, -5.5889454, -1.0621345, 1.0651581
6: 8.2564125, 10.0458202, 8.2744751, 10.0458050, -1.3649936, 1.3519273
7: -14.0097389, -12.1174345, -14.0097361, -12.1294889, -1.1561034, 1.1625247
8: -6.1156301, -4.6512346, -6.1093316, -4.6512423, -0.8973210, 0.8946238
9: -10.8449526, -8.5048056, -10.8447342, -8.5088797, -1.8527436, 1.8542905

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2130
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2130

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8089568
time: 3.78 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8089546
time: 6.63 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -1.6520326, 0.1124018, -1.6475005, 0.1262470, -1.5265756, 1.4985662
1: -17.9604244, -15.6270275, -17.9519997, -15.6137323, -1.5923023, 1.6026764
2: -6.5349836, -4.4760737, -6.5853863, -4.5002308, -1.4556661, 1.5352573
3: -13.9777927, -12.1096754, -14.0151386, -12.1148701, -1.3342991, 1.3656096
4: -5.6286044, -3.7163811, -5.6131539, -3.6883640, -1.8906698, 1.8402429
5: -7.0503912, -5.5889564, -7.0425291, -5.5800605, -1.0603986, 1.0942032
6: 8.2783871, 10.0458031, 8.3123417, 10.1139135, -1.4194174, 1.3209302
7: -14.0097370, -12.1223621, -14.0357504, -12.1219559, -1.1480534, 1.1811223
8: -6.1094365, -4.6512480, -6.1018839, -4.6322112, -0.8913603, 0.8826355
9: -10.8447075, -8.5121737, -10.8561497, -8.5267410, -1.8509269, 1.8477707

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2130
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2130

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8153171
time: 3.92 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8153157
time: 6.92 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 16.44 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 16.44
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8089568
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 16.44
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8089546
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 16.44
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8153171
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 16.44
Output dim: 6, lower bound: -0.8089547, upper bound: 0.8153157

## BFS IS instance: IS_B1_A1

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

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8002993, upper bound: 0.7974823
time: 4.11 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8011778, upper bound: 0.8052181
time: 3.61 seconds

## BFS IS instance: IS_B1_A2

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

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B1_A2_A1

### Relational analysis result of IS_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8002993, upper bound: 0.7974841
time: 4.01 seconds

## Relational analysis of IS_B1_A2_A2

### Relational analysis result of IS_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8011778, upper bound: 0.8052161
time: 4.15 seconds

## BFS IS instance: IS_B2_A1

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

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B2_A1_B1

### Relational analysis result of IS_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7949406, upper bound: 0.8067782
time: 4.15 seconds

## Relational analysis of IS_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8011776, upper bound: 0.8073669
time: 3.55 seconds

## BFS IS instance: IS_B2_A2

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

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_B2_A2_A1

### Relational analysis result of IS_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8002993, upper bound: 0.7949428
time: 5.15 seconds

## Relational analysis of IS_B2_A2_A2

### Relational analysis result of IS_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8011778, upper bound: 0.8011797
time: 3.68 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 14.47 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -0.8002993, upper bound: 0.7974823
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -0.8011778, upper bound: 0.8052181
IS_B1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -0.8002993, upper bound: 0.7974841
IS_B1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -0.8011778, upper bound: 0.8052161
IS_B2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -0.7949406, upper bound: 0.8067782
IS_B2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -0.8011776, upper bound: 0.8073669
IS_B2_A2_A1, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -0.8002993, upper bound: 0.7949428
IS_B2_A2_A2, status: Status.UNKNOWN, split count: 3, time: 14.47
Output dim: 6, lower bound: -0.8011778, upper bound: 0.8011797

## BFS IS instance: IS_B1_A1_A1

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

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8035681, upper bound: 0.7921510
time: 3.59 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7984172, upper bound: 0.7921512
time: 3.75 seconds

## BFS IS instance: IS_B1_A1_A2

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

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.8039793, upper bound: 0.7988354
time: 3.93 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7988334, upper bound: 0.7988351
time: 4.47 seconds

## BFS IS instance: IS_B1_A2_A1

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

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B1_A2_A1_B1

### Relational analysis result of IS_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7982987, upper bound: 0.7840479
time: 5.86 seconds

## Relational analysis of IS_B1_A2_A1_B2

### Relational analysis result of IS_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7840477
time: 7.97 seconds

## BFS IS instance: IS_B1_A2_A2

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

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B1_A2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7985337, upper bound: 0.7913169
time: 3.84 seconds

## Relational analysis of IS_B1_A2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7913152
time: 7.59 seconds

## BFS IS instance: IS_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6482308, 0.1124225, -1.6410147, 0.1308787, -1.5124202, 1.4873099
1: -17.9605503, -15.6416302, -17.9600925, -15.6518440, -1.5391593, 1.5514624
2: -6.5328522, -4.4835677, -6.5795999, -4.5005484, -1.4599857, 1.5110440
3: -13.9777975, -12.1258564, -14.0056801, -12.1403723, -1.2721095, 1.3109217
4: -5.6107421, -3.7176168, -5.5723267, -3.7033584, -1.8547382, 1.8052588
5: -7.0511942, -5.5977745, -7.0431242, -5.6016593, -1.0290442, 1.0211726
6: 8.2744865, 10.0319519, 8.3133726, 10.0787506, -1.3961425, 1.2836998
7: -14.0093460, -12.1316643, -14.0346870, -12.1278963, -1.1413140, 1.1779170
8: -6.1007776, -4.6512446, -6.0788412, -4.6320820, -0.8854077, 0.8515139
9: -10.8355474, -8.5088806, -10.8304892, -8.5228643, -1.8153176, 1.8234358

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B2_A1_B1_A1

### Relational analysis result of IS_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7840472, upper bound: 0.7982988
time: 4.27 seconds

## Relational analysis of IS_B2_A1_B1_A2

### Relational analysis result of IS_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7840472, upper bound: 0.7928541
time: 4.50 seconds

## BFS IS instance: IS_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6494200, 0.1124224, -1.6419480, 0.1262433, -1.5155401, 1.4894538
1: -17.9605503, -15.6300259, -17.9519997, -15.6275787, -1.5585966, 1.5600898
2: -6.5346189, -4.4835072, -6.5837455, -4.5005035, -1.4626894, 1.5158825
3: -13.9778023, -12.1171074, -14.0151215, -12.1173935, -1.2751908, 1.3243084
4: -5.6242318, -3.7165711, -5.6089945, -3.6892047, -1.8749514, 1.8108964
5: -7.0520096, -5.5907087, -7.0418663, -5.5878100, -1.0229168, 1.0448021
6: 8.2744780, 10.0435734, 8.3123579, 10.1027765, -1.3918858, 1.3097830
7: -14.0096779, -12.1300278, -14.0354652, -12.1244431, -1.1446414, 1.1792262
8: -6.1077771, -4.6512423, -6.0946827, -4.6322107, -0.8957323, 0.8568141
9: -10.8425074, -8.5088787, -10.8474274, -8.5267429, -1.8243566, 1.8407578

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B2_A1_B2_A1

### Relational analysis result of IS_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7913148, upper bound: 0.7985340
time: 4.30 seconds

## Relational analysis of IS_B2_A1_B2_A2

### Relational analysis result of IS_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7913148, upper bound: 0.7930825
time: 4.87 seconds

## BFS IS instance: IS_B2_A2_A1

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

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B2_A2_A1_B1

### Relational analysis result of IS_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7982987, upper bound: 0.7814038
time: 6.62 seconds

## Relational analysis of IS_B2_A2_A1_B2

### Relational analysis result of IS_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7814037
time: 4.77 seconds

## BFS IS instance: IS_B2_A2_A2

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

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B2_A2_A2_B1

### Relational analysis result of IS_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7985337, upper bound: 0.7871970
time: 6.10 seconds

## Relational analysis of IS_B2_A2_A2_B2

### Relational analysis result of IS_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7871970
time: 4.43 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 16.16 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.8035681, upper bound: 0.7921510
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7984172, upper bound: 0.7921512
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.8039793, upper bound: 0.7988354
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7988334, upper bound: 0.7988351
IS_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7982987, upper bound: 0.7840479
IS_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7840477
IS_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7985337, upper bound: 0.7913169
IS_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7913152
IS_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7840472, upper bound: 0.7982988
IS_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7840472, upper bound: 0.7928541
IS_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7913148, upper bound: 0.7985340
IS_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7913148, upper bound: 0.7930825
IS_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7982987, upper bound: 0.7814038
IS_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7814037
IS_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7985337, upper bound: 0.7871970
IS_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.16
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7871970

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6440289, 0.1170566, -1.6471694, 0.1085108, -1.4827547, 1.4937935
1: -17.9686394, -15.6648798, -17.9605465, -15.6540909, -1.5420084, 1.5210209
2: -6.5292134, -4.4837713, -6.5325465, -4.4856863, -1.4338522, 1.4393921
3: -13.9683418, -12.1419897, -13.9729261, -12.1260529, -1.2701378, 1.2594409
4: -5.5857277, -3.7300534, -5.5988836, -3.7188809, -1.8017902, 1.8030539
5: -7.0527544, -5.6105976, -7.0498624, -5.6035199, -1.0195487, 1.0176728
6: 8.2773838, 10.0093632, 8.2793694, 10.0257587, -1.3082552, 1.3033791
7: -14.0086727, -12.1354914, -14.0067625, -12.1324158, -1.1507957, 1.1425827
8: -6.0861721, -4.6511135, -6.0811892, -4.6513090, -0.8692393, 0.8617716
9: -10.8196745, -8.5050039, -10.8314877, -8.5175896, -1.8048129, 1.8303571

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B1_A1_A1_B1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7984172, upper bound: 0.7921512
time: 3.57 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7984172, upper bound: 0.7921512
time: 3.60 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -1.6432524, 0.1154664, -1.6752357, 0.1100100, -1.4807687, 1.5216064
1: -17.9686356, -15.6726389, -17.9867172, -15.6623955, -1.5391407, 1.5415175
2: -6.5290260, -4.4848266, -6.5332842, -4.4762630, -1.4360261, 1.4529881
3: -13.9629574, -12.1420927, -13.9640265, -12.1077604, -1.2858391, 1.2599778
4: -5.5770922, -3.7307401, -5.5909834, -3.6851740, -1.8319359, 1.7942772
5: -7.0520682, -5.6135392, -7.0698719, -5.6058974, -1.0307198, 1.0408137
6: 8.2800407, 10.0000200, 8.2517662, 10.0030651, -1.2989678, 1.3461528
7: -14.0024614, -12.1359663, -13.9913216, -12.1238098, -1.1677046, 1.1397233
8: -6.0780435, -4.6512089, -6.0826874, -4.5999532, -0.9238555, 0.8757362
9: -10.8173990, -8.5105639, -10.8628578, -8.5257874, -1.8334408, 1.8618546

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7920173, upper bound: 0.7818227
time: 3.82 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7882286, upper bound: 0.7819174
time: 4.18 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -1.6450552, 0.1124195, -1.6483579, 0.1085095, -1.4848390, 1.4969096
1: -17.9605503, -15.6407909, -17.9605484, -15.6425343, -1.5521011, 1.5403950
2: -6.5333314, -4.4837227, -6.5343180, -4.4856267, -1.4387760, 1.4420910
3: -13.9777908, -12.1190071, -13.9729319, -12.1173019, -1.2831240, 1.2625289
4: -5.6204309, -3.7172384, -5.6123734, -3.7178383, -1.8073921, 1.8233066
5: -7.0514894, -5.5967188, -7.0506821, -5.5964532, -1.0432210, 1.0115865
6: 8.2744904, 10.0354328, 8.2793617, 10.0373726, -1.3360033, 1.2991931
7: -14.0094528, -12.1319809, -14.0070915, -12.1307735, -1.1520946, 1.1457431
8: -6.1021070, -4.6512437, -6.0882020, -4.6513100, -0.8745893, 0.8720227
9: -10.8361607, -8.5088787, -10.8384552, -8.5175915, -1.8213415, 1.8393869

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7988334, upper bound: 0.7988354
time: 3.96 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7988334, upper bound: 0.7988354
time: 3.66 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -1.6442807, 0.1108269, -1.6763822, 0.1100103, -1.4828825, 1.5247035
1: -17.9605408, -15.6485720, -17.9867172, -15.6509590, -1.5494189, 1.5609031
2: -6.5331359, -4.4847784, -6.5350428, -4.4762034, -1.4409585, 1.4556599
3: -13.9724092, -12.1191092, -13.9640312, -12.0989962, -1.2988210, 1.2631040
4: -5.6117954, -3.7179570, -5.6044722, -3.6840956, -1.8374414, 1.8145580
5: -7.0507665, -5.5996580, -7.0706615, -5.5988040, -1.0544000, 1.0348059
6: 8.2771349, 10.0260878, 8.2517586, 10.0146914, -1.3266997, 1.3420839
7: -14.0032415, -12.1324415, -13.9916515, -12.1221561, -1.1689441, 1.1428754
8: -6.0939050, -4.6513386, -6.0897303, -4.5999527, -0.9292200, 0.8860106
9: -10.8338890, -8.5144348, -10.8697977, -8.5257835, -1.8499770, 1.8707008

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7923508, upper bound: 0.7884739
time: 3.95 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7885849, upper bound: 0.7885846
time: 3.67 seconds

## BFS IS instance: IS_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6471694, 0.1085108, -1.4828382, 1.5109811
1: -17.9600925, -15.6518440, -17.9605465, -15.6540909, -1.5399375, 1.5391524
2: -6.5795999, -4.5005484, -6.5325465, -4.4856863, -1.5042019, 1.4550066
3: -14.0056801, -12.1403723, -13.9729261, -12.1260529, -1.3106976, 1.2661510
4: -5.5723267, -3.7033584, -5.5988836, -3.7188809, -1.8040075, 1.8411579
5: -7.0431242, -5.6016593, -7.0498624, -5.6035199, -1.0119286, 1.0251970
6: 8.3133726, 10.0787506, 8.2793694, 10.0257587, -1.2686148, 1.3869967
7: -14.0346870, -12.1278963, -14.0067625, -12.1324158, -1.1759508, 1.1341877
8: -6.0788412, -4.6320820, -6.0811892, -4.6513090, -0.8514261, 0.8643287
9: -10.8304892, -8.5228643, -10.8314877, -8.5175896, -1.8063188, 1.8035808

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B1_A2_A1_B1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7840475
time: 6.64 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7840491
time: 4.03 seconds

## BFS IS instance: IS_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6402328, 0.1292862, -1.6752357, 0.1100100, -1.4808512, 1.5387950
1: -17.9600830, -15.6596031, -17.9867172, -15.6623955, -1.5370722, 1.5597031
2: -6.5794134, -4.5015931, -6.5332842, -4.4762630, -1.5063720, 1.4686165
3: -14.0002937, -12.1404762, -13.9640265, -12.1077604, -1.3263564, 1.2666860
4: -5.5636673, -3.7040591, -5.5909834, -3.6851740, -1.8341446, 1.8323922
5: -7.0424376, -5.6045985, -7.0698719, -5.6058974, -1.0231223, 1.0483189
6: 8.3159523, 10.0694008, 8.2517662, 10.0030651, -1.2594938, 1.4297733
7: -14.0284729, -12.1283579, -13.9913216, -12.1238098, -1.1928596, 1.1313667
8: -6.0706587, -4.6321712, -6.0826874, -4.5999532, -0.9061430, 0.8782974
9: -10.8281822, -8.5284052, -10.8628578, -8.5257874, -1.8348846, 1.8350797

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_B1_A2_A1_B2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7862604, upper bound: 0.7734508
time: 4.43 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7825571, upper bound: 0.7737607
time: 4.43 seconds

## BFS IS instance: IS_B1_A2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6483579, 0.1085095, -1.4849849, 1.5140991
1: -17.9519997, -15.6275787, -17.9605484, -15.6425343, -1.5486326, 1.5585909
2: -6.5837455, -4.5005035, -6.5343180, -4.4856267, -1.5090408, 1.4577031
3: -14.0151215, -12.1173935, -13.9729319, -12.1173019, -1.3240848, 1.2692394
4: -5.6089945, -3.6892047, -5.6123734, -3.7178383, -1.8096085, 1.8613653
5: -7.0418663, -5.5878100, -7.0506821, -5.5964532, -1.0355594, 1.0191115
6: 8.3123579, 10.1027765, 8.2793617, 10.0373726, -1.2946587, 1.3827381
7: -14.0354652, -12.1244431, -14.0070915, -12.1307735, -1.1772454, 1.1375153
8: -6.0946827, -4.6322107, -6.0882020, -4.6513100, -0.8567280, 0.8746696
9: -10.8474274, -8.5267429, -10.8384552, -8.5175915, -1.8236485, 1.8126097

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B1_A2_A2_B1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7913169
time: 4.53 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7913168
time: 4.64 seconds

## BFS IS instance: IS_B1_A2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6411656, 0.1246489, -1.6763822, 0.1100103, -1.4830379, 1.5418954
1: -17.9519939, -15.6353788, -17.9867172, -15.6509590, -1.5459528, 1.5791602
2: -6.5835495, -4.5015454, -6.5350428, -4.4762034, -1.5112190, 1.4712858
3: -14.0097332, -12.1174946, -13.9640312, -12.0989962, -1.3397365, 1.2698116
4: -5.6003361, -3.6899271, -5.6044722, -3.6840956, -1.8396502, 1.8526278
5: -7.0411434, -5.5907469, -7.0706615, -5.5988040, -1.0467613, 1.0423115
6: 8.3149233, 10.0934229, 8.2517586, 10.0146914, -1.2855225, 1.4256272
7: -14.0292521, -12.1248922, -13.9916515, -12.1221561, -1.1940944, 1.1346860
8: -6.0864878, -4.6323023, -6.0897303, -4.5999527, -0.9114373, 0.8886631
9: -10.8451242, -8.5322781, -10.8697977, -8.5257835, -1.8522277, 1.8439264

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_B1_A2_A2_B2_A1

### Relational analysis result of IS_B1_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7863690, upper bound: 0.7806372
time: 6.11 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2

### Relational analysis result of IS_B1_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7826910, upper bound: 0.7809453
time: 5.40 seconds

## BFS IS instance: IS_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6471694, 0.1085108, -1.6410147, 0.1308787, -1.5109811, 1.4828382
1: -17.9605465, -15.6540909, -17.9600925, -15.6518440, -1.5391526, 1.5399375
2: -6.5325465, -4.4856863, -6.5795999, -4.5005484, -1.4550066, 1.5042019
3: -13.9729261, -12.1260529, -14.0056801, -12.1403723, -1.2661510, 1.3106976
4: -5.5988836, -3.7188809, -5.5723267, -3.7033584, -1.8411579, 1.8040075
5: -7.0498624, -5.6035199, -7.0431242, -5.6016593, -1.0251970, 1.0119284
6: 8.2793694, 10.0257587, 8.3133726, 10.0787506, -1.3869967, 1.2686148
7: -14.0067625, -12.1324158, -14.0346870, -12.1278963, -1.1341877, 1.1759505
8: -6.0811892, -4.6513090, -6.0788412, -4.6320820, -0.8643286, 0.8514260
9: -10.8314877, -8.5175896, -10.8304892, -8.5228643, -1.8035808, 1.8063192

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B2_A1_B1_A1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7840472, upper bound: 0.7928538
time: 4.82 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7840472, upper bound: 0.7928540
time: 4.67 seconds

## BFS IS instance: IS_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6752357, 0.1100100, -1.6402328, 0.1292862, -1.5387950, 1.4808512
1: -17.9867172, -15.6623955, -17.9600830, -15.6596031, -1.5597029, 1.5370719
2: -6.5332842, -4.4762630, -6.5794134, -4.5015931, -1.4686165, 1.5063715
3: -13.9640265, -12.1077604, -14.0002937, -12.1404762, -1.2666860, 1.3263564
4: -5.5909834, -3.6851740, -5.5636673, -3.7040591, -1.8323927, 1.8341446
5: -7.0698719, -5.6058974, -7.0424376, -5.6045985, -1.0483189, 1.0231224
6: 8.2517662, 10.0030651, 8.3159523, 10.0694008, -1.4297733, 1.2594938
7: -13.9913216, -12.1238098, -14.0284729, -12.1283579, -1.1313672, 1.1928599
8: -6.0826874, -4.5999532, -6.0706587, -4.6321712, -0.8782974, 0.9061430
9: -10.8628578, -8.5257874, -10.8281822, -8.5284052, -1.8350797, 1.8348851

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: B, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_B2_A1_B1_A2_B1

### Relational analysis result of IS_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7734487, upper bound: 0.7862608
time: 6.62 seconds

## Relational analysis of IS_B2_A1_B1_A2_B2

### Relational analysis result of IS_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7737608, upper bound: 0.7825574
time: 4.46 seconds

## BFS IS instance: IS_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6483579, 0.1085095, -1.6419480, 0.1262433, -1.5140991, 1.4849849
1: -17.9605484, -15.6425343, -17.9519997, -15.6275787, -1.5585909, 1.5486326
2: -6.5343180, -4.4856267, -6.5837455, -4.5005035, -1.4577031, 1.5090408
3: -13.9729319, -12.1173019, -14.0151215, -12.1173935, -1.2692394, 1.3240848
4: -5.6123734, -3.7178383, -5.6089945, -3.6892047, -1.8613653, 1.8096085
5: -7.0506821, -5.5964532, -7.0418663, -5.5878100, -1.0191114, 1.0355594
6: 8.2793617, 10.0373726, 8.3123579, 10.1027765, -1.3827381, 1.2946589
7: -14.0070915, -12.1307735, -14.0354652, -12.1244431, -1.1375155, 1.1772454
8: -6.0882020, -4.6513100, -6.0946827, -4.6322107, -0.8746694, 0.8567281
9: -10.8384552, -8.5175915, -10.8474274, -8.5267429, -1.8126101, 1.8236480

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_B2_A1_B2_A1_B1

### Relational analysis result of IS_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7913148, upper bound: 0.7928535
time: 6.60 seconds

## Relational analysis of IS_B2_A1_B2_A1_B2

### Relational analysis result of IS_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7913148, upper bound: 0.7930826
time: 4.97 seconds

## BFS IS instance: IS_B2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6763822, 0.1100103, -1.6411656, 0.1246489, -1.5418954, 1.4830379
1: -17.9867172, -15.6509590, -17.9519939, -15.6353788, -1.5791602, 1.5459528
2: -6.5350428, -4.4762034, -6.5835495, -4.5015454, -1.4712858, 1.5112185
3: -13.9640312, -12.0989962, -14.0097332, -12.1174946, -1.2698116, 1.3397365
4: -5.6044722, -3.6840956, -5.6003361, -3.6899271, -1.8526278, 1.8396497
5: -7.0706615, -5.5988040, -7.0411434, -5.5907469, -1.0423114, 1.0467613
6: 8.2517586, 10.0146914, 8.3149233, 10.0934229, -1.4256277, 1.2855227
7: -13.9916515, -12.1221561, -14.0292521, -12.1248922, -1.1346860, 1.1940944
8: -6.0897303, -4.5999527, -6.0864878, -4.6323023, -0.8886631, 0.9114373
9: -10.8697977, -8.5257835, -10.8451242, -8.5322781, -1.8439264, 1.8522272

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: B, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_B2_A1_B2_A2_B1

### Relational analysis result of IS_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7806367, upper bound: 0.7863690
time: 6.51 seconds

## Relational analysis of IS_B2_A1_B2_A2_B2

### Relational analysis result of IS_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7809432, upper bound: 0.7826905
time: 4.24 seconds

## BFS IS instance: IS_B2_A2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6440692, 0.1223218, -1.5054965, 1.5165710
1: -17.9600925, -15.6518440, -17.9519939, -15.6410599, -1.5803008, 1.5597630
2: -6.5795999, -4.5005484, -6.5829601, -4.5024443, -1.4482193, 1.4537559
3: -14.0056801, -12.1403723, -14.0102539, -12.1244354, -1.2957630, 1.2845931
4: -5.5723267, -3.7033584, -5.5862617, -3.6908498, -1.8004208, 1.8016834
5: -7.0431242, -5.6016593, -7.0402403, -5.5946302, -1.0561697, 1.0558295
6: 8.3133726, 10.0787506, 8.3170662, 10.0943298, -1.2947614, 1.2902203
7: -14.0346870, -12.1278963, -14.0327730, -12.1248474, -1.1665163, 1.1583219
8: -6.0788412, -4.6320820, -6.0737958, -4.6322742, -0.8559766, 0.8485144
9: -10.8304892, -8.5228643, -10.8426228, -8.5354233, -1.8026371, 1.8281970

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B2_A2_A1_B1_A1

### Relational analysis result of IS_B2_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7814049
time: 4.67 seconds

## Relational analysis of IS_B2_A2_A1_B1_A2

### Relational analysis result of IS_B2_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7814054
time: 3.86 seconds

## BFS IS instance: IS_B2_A2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6402328, 0.1292862, -1.6718475, 0.1238356, -1.5035601, 1.5440431
1: -17.9600830, -15.6596031, -17.9781666, -15.6494064, -1.5775847, 1.5804436
2: -6.5794134, -4.5015931, -6.5836935, -4.4930201, -1.4502068, 1.4673300
3: -14.0002937, -12.1404762, -14.0013189, -12.1061459, -1.3116007, 1.2851968
4: -5.5636673, -3.7040591, -5.5781975, -3.6571569, -1.8305626, 1.7929454
5: -7.0424376, -5.6045985, -7.0602121, -5.5968885, -1.0673859, 1.0789146
6: 8.3159523, 10.0694008, 8.2899590, 10.0716133, -1.2853847, 1.3324704
7: -14.0284729, -12.1283579, -14.0173416, -12.1163216, -1.1837656, 1.1555123
8: -6.0706587, -4.6321712, -6.0754299, -4.5808887, -0.9106629, 0.8622031
9: -10.8281822, -8.5284052, -10.8737812, -8.5435486, -1.8312631, 1.8596048

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_B2_A2_A1_B2_A1

### Relational analysis result of IS_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7862604, upper bound: 0.7708645
time: 4.08 seconds

## Relational analysis of IS_B2_A2_A1_B2_A2

### Relational analysis result of IS_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7825571, upper bound: 0.7711559
time: 4.30 seconds

## BFS IS instance: IS_B2_A2_A2_B1

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6452184, 0.1223222, -1.5076418, 1.5197124
1: -17.9519997, -15.6275787, -17.9519958, -15.6292820, -1.5910568, 1.5782509
2: -6.5837455, -4.5005035, -6.5847278, -4.5023832, -1.4530501, 1.4563823
3: -14.0151215, -12.1173935, -14.0102606, -12.1156864, -1.3075061, 1.2885208
4: -5.6089945, -3.6892047, -5.6003675, -3.6898184, -1.8060265, 1.8219538
5: -7.0418663, -5.5878100, -7.0410614, -5.5875406, -1.0813236, 1.0480316
6: 8.3123579, 10.1027765, 8.3170605, 10.1053095, -1.3228788, 1.2857571
7: -14.0354652, -12.1244431, -14.0331030, -12.1232271, -1.1680803, 1.1616290
8: -6.0946827, -4.6322107, -6.0807629, -4.6322737, -0.8613245, 0.8587915
9: -10.8474274, -8.5267429, -10.8497868, -8.5354223, -1.8191652, 1.8372211

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B2_A2_A2_B1_A1

### Relational analysis result of IS_B2_A2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7871954
time: 4.13 seconds

## Relational analysis of IS_B2_A2_A2_B1_A2

### Relational analysis result of IS_B2_A2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7871955
time: 4.30 seconds

## BFS IS instance: IS_B2_A2_A2_B2

### Backsubstitution after applying IS history:
0: -1.6411656, 0.1246489, -1.6729534, 0.1238354, -1.5057259, 1.5471654
1: -17.9519939, -15.6353788, -17.9781685, -15.6378164, -1.5885282, 1.5989475
2: -6.5835495, -4.5015454, -6.5854564, -4.4929590, -1.4550467, 1.4699316
3: -14.0097332, -12.1174946, -14.0013256, -12.0973797, -1.3233376, 1.2891788
4: -5.6003361, -3.6899271, -5.5923057, -3.6560876, -1.8360720, 1.8132463
5: -7.0411434, -5.5907469, -7.0610008, -5.5897741, -1.0925474, 1.0711958
6: 8.3149233, 10.0934229, 8.2899513, 10.0826025, -1.3134892, 1.3281224
7: -14.0292521, -12.1248922, -14.0176697, -12.1146898, -1.1852832, 1.1588111
8: -6.0864878, -4.6323023, -6.0824299, -4.5808868, -0.9160032, 0.8725492
9: -10.8451242, -8.5322781, -10.8809109, -8.5435467, -1.8477988, 1.8684473

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_B2_A2_A2_B2_A1

### Relational analysis result of IS_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7863690, upper bound: 0.7765644
time: 4.49 seconds

## Relational analysis of IS_B2_A2_A2_B2_A2

### Relational analysis result of IS_B2_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7826910, upper bound: 0.7768447
time: 4.36 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 14.52 seconds
IS_B1_A1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7984172, upper bound: 0.7921512
IS_B1_A1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7984172, upper bound: 0.7921512
IS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7920173, upper bound: 0.7818227
IS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7882286, upper bound: 0.7819174
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7988334, upper bound: 0.7988354
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7988334, upper bound: 0.7988354
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7923508, upper bound: 0.7884739
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7885849, upper bound: 0.7885846
IS_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7840475
IS_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7840491
IS_B1_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7862604, upper bound: 0.7734508
IS_B1_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7825571, upper bound: 0.7737607
IS_B1_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7913169
IS_B1_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7913168
IS_B1_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7863690, upper bound: 0.7806372
IS_B1_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7826910, upper bound: 0.7809453
IS_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7840472, upper bound: 0.7928538
IS_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7840472, upper bound: 0.7928540
IS_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7734487, upper bound: 0.7862608
IS_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7737608, upper bound: 0.7825574
IS_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7913148, upper bound: 0.7928535
IS_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7913148, upper bound: 0.7930826
IS_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7806367, upper bound: 0.7863690
IS_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7809432, upper bound: 0.7826905
IS_B2_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7814049
IS_B2_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7928539, upper bound: 0.7814054
IS_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7862604, upper bound: 0.7708645
IS_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7825571, upper bound: 0.7711559
IS_B2_A2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7871954
IS_B2_A2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7930827, upper bound: 0.7871955
IS_B2_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7863690, upper bound: 0.7765644
IS_B2_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.52
Output dim: 6, lower bound: -0.7826910, upper bound: 0.7768447

## BFS IS instance: IS_B1_A1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6429667, 0.1131482, -1.6471694, 0.1085108, -1.4813156, 1.4893250
1: -17.9686394, -15.6773357, -17.9605465, -15.6540909, -1.5420027, 1.5094833
2: -6.5289249, -4.4858875, -6.5325465, -4.4856863, -1.4288974, 1.4325504
3: -13.9634705, -12.1421843, -13.9729261, -12.1260529, -1.2641859, 1.2592173
4: -5.5738678, -3.7312579, -5.5988836, -3.7188809, -1.7882099, 1.8017578
5: -7.0514927, -5.6163478, -7.0498624, -5.6035199, -1.0157399, 1.0084223
6: 8.2822876, 10.0031853, 8.2793694, 10.0257587, -1.2991080, 1.2883272
7: -14.0060873, -12.1362534, -14.0067625, -12.1324158, -1.1436698, 1.1406159
8: -6.0666018, -4.6511798, -6.0811892, -4.6513090, -0.8481016, 0.8616854
9: -10.8155832, -8.5137215, -10.8314877, -8.5175896, -1.7931046, 1.8132482

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_B1_A1_A1_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7974454, upper bound: 0.7816230
time: 3.94 seconds

## Relational analysis of IS_B1_A1_A1_B1_A1_A2

### Relational analysis result of IS_B1_A1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7925592, upper bound: 0.7818490
time: 3.90 seconds

## BFS IS instance: IS_B1_A1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6710336, 0.1146619, -1.6471694, 0.1085108, -1.5092897, 1.4888997
1: -17.9948082, -15.6855450, -17.9605465, -15.6540909, -1.5703406, 1.5049081
2: -6.5296888, -4.4764652, -6.5325465, -4.4856863, -1.4300799, 1.4393139
3: -13.9545717, -12.1238918, -13.9729261, -12.1260529, -1.2584438, 1.2790904
4: -5.5659685, -3.6974487, -5.5988836, -3.7188809, -1.7745161, 1.8405523
5: -7.0716925, -5.6187625, -7.0498624, -5.6035199, -1.0410953, 1.0111287
6: 8.2547817, 9.9804878, 8.2793694, 10.0257587, -1.3457260, 1.2935123
7: -13.9906483, -12.1276608, -14.0067625, -12.1324158, -1.1421649, 1.1604903
8: -6.0681472, -4.5998230, -6.0811892, -4.6513090, -0.8556511, 0.9230543
9: -10.8470211, -8.5219250, -10.8314877, -8.5175896, -1.8334336, 1.8056169

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_B1_A1_A1_B1_A2_B1

### Relational analysis result of IS_B1_A1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7920186, upper bound: 0.7857865
time: 3.96 seconds

## Relational analysis of IS_B1_A1_A1_B1_A2_B2

### Relational analysis result of IS_B1_A1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7925592, upper bound: 0.7819152
time: 4.07 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6456702, 0.1061088, -1.6746283, 0.1067790, -1.4770494, 1.5057845
1: -17.9722099, -15.7078743, -17.9866734, -15.6743984, -1.5268250, 1.5053093
2: -6.5256386, -4.4804568, -6.5321455, -4.4770374, -1.4283319, 1.4523497
3: -13.9701338, -12.1512070, -13.9636040, -12.1108341, -1.2859197, 1.2479744
4: -5.5204930, -3.7460003, -5.5708370, -3.6857071, -1.7796288, 1.7470803
5: -7.0626364, -5.6225076, -7.0691161, -5.6089149, -1.0344751, 1.0272191
6: 8.2978840, 10.0147047, 8.2586994, 10.0030251, -1.2710600, 1.3374338
7: -14.0013580, -12.1366625, -13.9909163, -12.1240339, -1.1664615, 1.1382284
8: -6.0730200, -4.6709099, -6.0812349, -4.6071749, -0.9076183, 0.8515551
9: -10.7982883, -8.5164738, -10.8561506, -8.5261240, -1.8147016, 1.8488021

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B1_A1_A1_B2_A1_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7920173, upper bound: 0.7816231
time: 3.94 seconds

## Relational analysis of IS_B1_A1_A1_B2_A1_A2

### Relational analysis result of IS_B1_A1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7920173, upper bound: 0.7818225
time: 4.64 seconds

## BFS IS instance: IS_B1_A1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6421720, 0.1070712, -1.6749296, 0.1077275, -1.4731121, 1.5169082
1: -17.9683704, -15.6928587, -17.9866486, -15.6680317, -1.5297856, 1.5288489
2: -6.5190010, -4.4862652, -6.5305996, -4.4766474, -1.4289479, 1.4503274
3: -13.9621639, -12.1539593, -13.9638262, -12.1109200, -1.2828431, 1.2539730
4: -5.5699110, -3.7316232, -5.5891995, -3.6854539, -1.7814789, 1.7770061
5: -7.0508418, -5.6310434, -7.0695066, -5.6105685, -1.0268433, 1.0356650
6: 8.3035583, 9.9999409, 8.2579145, 10.0030479, -1.2744200, 1.3405671
7: -14.0017185, -12.1363726, -13.9911346, -12.1239376, -1.1674352, 1.1385078
8: -6.0756931, -4.6585174, -6.0819292, -4.6020169, -0.9169593, 0.8615003
9: -10.8144350, -8.5111084, -10.8620586, -8.5259628, -1.8376613, 1.8501701

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B1_A1_A1_B2_A2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7881159, upper bound: 0.7818471
time: 3.94 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7881159, upper bound: 0.7819151
time: 3.94 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6439968, 0.1085086, -1.6483579, 0.1085095, -1.4833827, 1.4924378
1: -17.9605465, -15.6532793, -17.9605484, -15.6425343, -1.5520949, 1.5288744
2: -6.5330286, -4.4858408, -6.5343180, -4.4856267, -1.4337921, 1.4352493
3: -13.9729214, -12.1192026, -13.9729319, -12.1173019, -1.2771668, 1.2623057
4: -5.6085730, -3.7185049, -5.6123734, -3.7178383, -1.7938118, 1.8220677
5: -7.0501585, -5.6024623, -7.0506821, -5.5964532, -1.0393839, 1.0023351
6: 8.2793722, 10.0292358, 8.2793617, 10.0373726, -1.3268580, 1.2840974
7: -14.0068674, -12.1327229, -14.0070915, -12.1307735, -1.1449692, 1.1437697
8: -6.0825377, -4.6513100, -6.0882020, -4.6513100, -0.8535141, 0.8719351
9: -10.8320770, -8.5175905, -10.8384552, -8.5175915, -1.8095970, 1.8222694

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_B1_A1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7923587, upper bound: 0.7923528
time: 3.86 seconds

## Relational analysis of IS_B1_A1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7929188, upper bound: 0.7884616
time: 4.06 seconds

## BFS IS instance: IS_B1_A1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6720339, 0.1100070, -1.6483579, 0.1085095, -1.5113988, 1.4919772
1: -17.9867172, -15.6616745, -17.9605484, -15.6425343, -1.5804286, 1.5243454
2: -6.5337782, -4.4764175, -6.5343180, -4.4856267, -1.4349155, 1.4420114
3: -13.9640217, -12.1008959, -13.9729319, -12.1173019, -1.2713766, 1.2821794
4: -5.6006718, -3.6847847, -5.6123734, -3.7178383, -1.7801170, 1.8609891
5: -7.0701571, -5.6048150, -7.0506821, -5.5964532, -1.0646381, 1.0050421
6: 8.2517681, 10.0065508, 8.2793617, 10.0373726, -1.3733768, 1.2893434
7: -13.9914274, -12.1240950, -14.0070915, -12.1307735, -1.1434639, 1.1635633
8: -6.0840840, -4.5999527, -6.0882020, -4.6513100, -0.8610085, 0.9332891
9: -10.8633537, -8.5257845, -10.8384552, -8.5175915, -1.8494568, 1.8146205

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_B1_A1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7923587, upper bound: 0.7923505
time: 3.90 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7929188, upper bound: 0.7885846
time: 3.99 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6467400, 0.1014740, -1.6757712, 0.1067795, -1.4792190, 1.5088882
1: -17.9641151, -15.6839485, -17.9866734, -15.6630087, -1.5371413, 1.5245845
2: -6.5297480, -4.4804068, -6.5339069, -4.4769735, -1.4332695, 1.4550037
3: -13.9795847, -12.1282206, -13.9636106, -12.1020660, -1.2988987, 1.2510962
4: -5.5551987, -3.7333541, -5.5843267, -3.6846299, -1.7851238, 1.7675123
5: -7.0613933, -5.6086078, -7.0699048, -5.6018209, -1.0580518, 1.0212387
6: 8.2949400, 10.0408049, 8.2586918, 10.0146484, -1.2987676, 1.3334222
7: -14.0021400, -12.1331387, -13.9912453, -12.1223812, -1.1676955, 1.1413689
8: -6.0889006, -4.6710415, -6.0882874, -4.6071730, -0.9131751, 0.8618617
9: -10.8147831, -8.5203371, -10.8630896, -8.5261221, -1.8312407, 1.8576365

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B1_A1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7857865, upper bound: 0.7881233
time: 4.14 seconds

## Relational analysis of IS_B1_A1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7857865, upper bound: 0.7881233
time: 4.95 seconds

## BFS IS instance: IS_B1_A1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6432015, 0.1024354, -1.6760744, 0.1077296, -1.4752483, 1.5200419
1: -17.9602776, -15.6685772, -17.9866447, -15.6565552, -1.5400853, 1.5480912
2: -6.5231123, -4.4862146, -6.5323586, -4.4765859, -1.4339199, 1.4530077
3: -13.9716158, -12.1309719, -13.9638319, -12.1021557, -1.2958207, 1.2570982
4: -5.6046157, -3.7189045, -5.6026878, -3.6843765, -1.7870264, 1.7973075
5: -7.0494928, -5.6171503, -7.0702939, -5.6034756, -1.0504768, 1.0296615
6: 8.3006325, 10.0260086, 8.2579060, 10.0146723, -1.3021779, 1.3365011
7: -14.0024977, -12.1328497, -13.9914656, -12.1222820, -1.1686704, 1.1416514
8: -6.0915718, -4.6586456, -6.0889769, -4.6020160, -0.9223175, 0.8718811
9: -10.8309250, -8.5149765, -10.8689966, -8.5259619, -1.8540659, 1.8590097

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B1_A1_A2_B2_A2_B1

### Relational analysis result of IS_B1_A1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7819153, upper bound: 0.7882282
time: 3.92 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7819153, upper bound: 0.7885849
time: 4.57 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6399505, 0.1269582, -1.6471694, 0.1085108, -1.4813662, 1.5065188
1: -17.9600868, -15.6643028, -17.9605465, -15.6540909, -1.5399327, 1.5277433
2: -6.5793109, -4.5026445, -6.5325465, -4.4856863, -1.4992380, 1.4482055
3: -14.0008049, -12.1405697, -13.9729261, -12.1260529, -1.3046393, 1.2659235
4: -5.5604362, -3.7045887, -5.5988836, -3.7188809, -1.7903996, 1.8398819
5: -7.0418634, -5.6074033, -7.0498624, -5.6035199, -1.0081594, 1.0158199
6: 8.3181076, 10.0725622, 8.2793694, 10.0257587, -1.2597871, 1.3719401
7: -14.0321026, -12.1286354, -14.0067625, -12.1324158, -1.1688259, 1.1322775
8: -6.0592737, -4.6321430, -6.0811892, -4.6513090, -0.8304543, 0.8642459
9: -10.8263445, -8.5315514, -10.8314877, -8.5175896, -1.7945008, 1.7864747

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_B1_A2_A1_B1_A1_A1

### Relational analysis result of IS_B1_A2_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7918980, upper bound: 0.7733263
time: 4.49 seconds

## Relational analysis of IS_B1_A2_A1_B1_A1_A2

### Relational analysis result of IS_B1_A2_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7872455, upper bound: 0.7736843
time: 4.52 seconds

## BFS IS instance: IS_B1_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6677475, 0.1284892, -1.6471694, 0.1085108, -1.5090051, 1.5061269
1: -17.9862614, -15.6726294, -17.9605465, -15.6540909, -1.5682797, 1.5231044
2: -6.5800805, -4.4932203, -6.5325465, -4.4856863, -1.5003991, 1.4549203
3: -13.9918699, -12.1222744, -13.9729261, -12.1260529, -1.2989707, 1.2857475
4: -5.5523729, -3.6708283, -5.5988836, -3.7188809, -1.7766571, 1.8787289
5: -7.0620794, -5.6097002, -7.0498624, -5.6035199, -1.0336118, 1.0186625
6: 8.2910995, 10.0498381, 8.2793694, 10.0257587, -1.3062901, 1.3771319
7: -14.0166693, -12.1201248, -14.0067625, -12.1324158, -1.1673338, 1.1524911
8: -6.0609593, -4.5807586, -6.0811892, -4.6513090, -0.8380127, 0.9256428
9: -10.8575659, -8.5396843, -10.8314877, -8.5175896, -1.8344846, 1.7788458

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_B1_A2_A1_B1_A2_B1

### Relational analysis result of IS_B1_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7864701, upper bound: 0.7776299
time: 6.80 seconds

## Relational analysis of IS_B1_A2_A1_B1_A2_B2

### Relational analysis result of IS_B1_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7872455, upper bound: 0.7737628
time: 6.10 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6423986, 0.1199467, -1.6746283, 0.1067790, -1.4769197, 1.5229564
1: -17.9636612, -15.6950607, -17.9866734, -15.6743984, -1.5247221, 1.5231645
2: -6.5760260, -4.4971857, -6.5321455, -4.4770374, -1.4986777, 1.4679480
3: -14.0075321, -12.1495895, -13.9636040, -12.1108341, -1.3262749, 1.2546654
4: -5.5071721, -3.7193534, -5.5708370, -3.6857071, -1.7819214, 1.7852225
5: -7.0530348, -5.6135206, -7.0691161, -5.6089149, -1.0269141, 1.0349764
6: 8.3335619, 10.0840874, 8.2586994, 10.0030251, -1.2318192, 1.4210467
7: -14.0273771, -12.1290169, -13.9909163, -12.1240339, -1.1916299, 1.1299164
8: -6.0655575, -4.6518707, -6.0812349, -4.6071749, -0.8895857, 0.8541272
9: -10.8091850, -8.5342703, -10.8561506, -8.5261240, -1.8162074, 1.8220325

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B1_A2_A1_B2_A1_A1

### Relational analysis result of IS_B1_A2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7862604, upper bound: 0.7733263
time: 6.96 seconds

## Relational analysis of IS_B1_A2_A1_B2_A1_A2

### Relational analysis result of IS_B1_A2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7862604, upper bound: 0.7734486
time: 5.04 seconds

## BFS IS instance: IS_B1_A2_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6391398, 0.1209112, -1.6749296, 0.1077275, -1.4729528, 1.5341139
1: -17.9598217, -15.6788321, -17.9866486, -15.6680317, -1.5276847, 1.5466471
2: -6.5693884, -4.5030036, -6.5305996, -4.4766474, -1.4992948, 1.4659619
3: -13.9994755, -12.1523428, -13.9638262, -12.1109200, -1.3233075, 1.2606754
4: -5.5564585, -3.7049608, -5.5891995, -3.6854539, -1.7837400, 1.8151350
5: -7.0412178, -5.6216121, -7.0695066, -5.6105685, -1.0192933, 1.0434725
6: 8.3396854, 10.0693207, 8.2579145, 10.0030479, -1.2349875, 1.4241834
7: -14.0277328, -12.1287346, -13.9911346, -12.1239376, -1.1926017, 1.1301796
8: -6.0683994, -4.6394787, -6.0819292, -4.6020169, -0.8988907, 0.8640716
9: -10.8252678, -8.5289240, -10.8620586, -8.5259628, -1.8391614, 1.8233981

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_B1_A2_A1_B2_A2_A1

### Relational analysis result of IS_B1_A2_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7824221, upper bound: 0.7736824
time: 6.78 seconds

## Relational analysis of IS_B1_A2_A1_B2_A2_A2

### Relational analysis result of IS_B1_A2_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7824221, upper bound: 0.7737614
time: 7.01 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6408842, 0.1223198, -1.6483579, 0.1085095, -1.4835186, 1.5096335
1: -17.9519939, -15.6401024, -17.9605484, -15.6425343, -1.5486269, 1.5471938
2: -6.5834417, -4.5025983, -6.5343180, -4.4856267, -1.5040503, 1.4509020
3: -14.0102463, -12.1175861, -13.9729319, -12.1173019, -1.3180132, 1.2690115
4: -5.5971012, -3.6904771, -5.6123734, -3.7178383, -1.7960014, 1.8601437
5: -7.0405369, -5.5935493, -7.0506821, -5.5964532, -1.0317616, 1.0097337
6: 8.3170710, 10.0965672, 8.2793617, 10.0373726, -1.2858324, 1.3676333
7: -14.0328798, -12.1251659, -14.0070915, -12.1307735, -1.1701200, 1.1355987
8: -6.0751190, -4.6322737, -6.0882020, -4.6513100, -0.8357919, 0.8745868
9: -10.8432903, -8.5354204, -10.8384552, -8.5175915, -1.8117971, 1.7954960

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_B1_A2_A2_B1_A1_A1

### Relational analysis result of IS_B1_A2_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7920104, upper bound: 0.7804739
time: 7.98 seconds

## Relational analysis of IS_B1_A2_A2_B1_A1_A2

### Relational analysis result of IS_B1_A2_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7874441, upper bound: 0.7808199
time: 5.17 seconds

## BFS IS instance: IS_B1_A2_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6686337, 0.1238341, -1.6483579, 0.1085095, -1.5111918, 1.5092235
1: -17.9781666, -15.6485596, -17.9605484, -15.6425343, -1.5769687, 1.5425940
2: -6.5841751, -4.4931755, -6.5343180, -4.4856267, -1.5051556, 1.4576159
3: -14.0013113, -12.0992775, -13.9729319, -12.1173019, -1.3122892, 1.2888360
4: -5.5890379, -3.6567700, -5.6123734, -3.7178383, -1.7822571, 1.8991203
5: -7.0604973, -5.5957823, -7.0506821, -5.5964532, -1.0571132, 1.0125769
6: 8.2899628, 10.0738573, 8.2793617, 10.0373726, -1.3322349, 1.3728886
7: -14.0174465, -12.1166172, -14.0070915, -12.1307735, -1.1686275, 1.1557460
8: -6.0768023, -4.5808878, -6.0882020, -4.6513100, -0.8432869, 0.9359891
9: -10.8743534, -8.5435457, -10.8384552, -8.5175915, -1.8513126, 1.7878504

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: B, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_B1_A2_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7866113, upper bound: 0.7847730
time: 4.44 seconds

## Relational analysis of IS_B1_A2_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7874441, upper bound: 0.7809453
time: 4.86 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6433651, 0.1153165, -1.6757712, 0.1067795, -1.4791698, 1.5260634
1: -17.9555683, -15.6710224, -17.9866734, -15.6630087, -1.5336394, 1.5424211
2: -6.5801616, -4.4971414, -6.5339069, -4.4769735, -1.5035276, 1.4705982
3: -14.0169725, -12.1266050, -13.9636106, -12.1020660, -1.3396640, 1.2577877
4: -5.5438395, -3.7053399, -5.5843267, -3.6846299, -1.7874146, 1.8056064
5: -7.0517821, -5.5996494, -7.0699048, -5.6018209, -1.0504487, 1.0289963
6: 8.3324938, 10.1081419, 8.2586918, 10.0146484, -1.2578225, 1.4169583
7: -14.0281582, -12.1255531, -13.9912453, -12.1223812, -1.1928601, 1.1332459
8: -6.0813489, -4.6520023, -6.0882874, -4.6071730, -0.8951402, 0.8645259
9: -10.8261318, -8.5381355, -10.8630896, -8.5261221, -1.8335505, 1.8308668

Time for backsubstitution: 5.52 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B1_A2_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7811120, upper bound: 0.7803630
time: 5.46 seconds

## Relational analysis of IS_B1_A2_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7811120, upper bound: 0.7806367
time: 5.44 seconds

## BFS IS instance: IS_B1_A2_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6400716, 0.1162765, -1.6760744, 0.1077296, -1.4751935, 1.5372510
1: -17.9517288, -15.6544418, -17.9866447, -15.6565552, -1.5365844, 1.5659037
2: -6.5735240, -4.5029583, -6.5323586, -4.4765859, -1.5041809, 1.4686394
3: -14.0089140, -12.1293583, -13.9638319, -12.1021557, -1.3366914, 1.2637992
4: -5.5931263, -3.6908839, -5.6026878, -3.6843765, -1.7892876, 1.8353887
5: -7.0398722, -5.6077471, -7.0702939, -5.6034756, -1.0428851, 1.0374696
6: 8.3386335, 10.0933437, 8.2579060, 10.0146723, -1.2610407, 1.4200416
7: -14.0285149, -12.1252718, -13.9914656, -12.1222820, -1.1938312, 1.1335022
8: -6.0842462, -4.6396089, -6.0889769, -4.6020160, -0.9041831, 0.8745449
9: -10.8422117, -8.5327911, -10.8689966, -8.5259619, -1.8563700, 1.8322382

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_B1_A2_A2_B2_A2_B1

### Relational analysis result of IS_B1_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7774023, upper bound: 0.7806728
time: 4.37 seconds

## Relational analysis of IS_B1_A2_A2_B2_A2_B2

### Relational analysis result of IS_B1_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7774023, upper bound: 0.7809433
time: 4.39 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6471694, 0.1085108, -1.6399505, 0.1269582, -1.5065188, 1.4813666
1: -17.9605465, -15.6540909, -17.9600868, -15.6643028, -1.5277433, 1.5399327
2: -6.5325465, -4.4856863, -6.5793109, -4.5026445, -1.4482055, 1.4992380
3: -13.9729261, -12.1260529, -14.0008049, -12.1405697, -1.2659235, 1.3046393
4: -5.5988836, -3.7188809, -5.5604362, -3.7045887, -1.8398819, 1.7903996
5: -7.0498624, -5.6035199, -7.0418634, -5.6074033, -1.0158200, 1.0081594
6: 8.2793694, 10.0257587, 8.3181076, 10.0725622, -1.3719404, 1.2597871
7: -14.0067625, -12.1324158, -14.0321026, -12.1286354, -1.1322777, 1.1688259
8: -6.0811892, -4.6513090, -6.0592737, -4.6321430, -0.8642457, 0.8304541
9: -10.8314877, -8.5175896, -10.8263445, -8.5315514, -1.7864752, 1.7945013

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_B2_A1_B1_A1_B1_B1

### Relational analysis result of IS_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7733262, upper bound: 0.7918977
time: 5.54 seconds

## Relational analysis of IS_B2_A1_B1_A1_B1_B2

### Relational analysis result of IS_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7736823, upper bound: 0.7812634
time: 9.70 seconds

## BFS IS instance: IS_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.6471694, 0.1085108, -1.6677475, 0.1284892, -1.5061269, 1.5090051
1: -17.9605465, -15.6540909, -17.9862614, -15.6726294, -1.5231047, 1.5682797
2: -6.5325465, -4.4856863, -6.5800805, -4.4932203, -1.4549203, 1.5003996
3: -13.9729261, -12.1260529, -13.9918699, -12.1222744, -1.2857475, 1.2989707
4: -5.5988836, -3.7188809, -5.5523729, -3.6708283, -1.8787289, 1.7766566
5: -7.0498624, -5.6035199, -7.0620794, -5.6097002, -1.0186625, 1.0336119
6: 8.2793694, 10.0257587, 8.2910995, 10.0498381, -1.3771319, 1.3062899
7: -14.0067625, -12.1324158, -14.0166693, -12.1201248, -1.1524913, 1.1673336
8: -6.0811892, -4.6513090, -6.0609593, -4.5807586, -0.9256430, 0.8380128
9: -10.8314877, -8.5175896, -10.8575659, -8.5396843, -1.7788458, 1.8344846

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7776279, upper bound: 0.7864699
time: 6.75 seconds

## Relational analysis of IS_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7736823, upper bound: 0.7872457
time: 5.04 seconds

## BFS IS instance: IS_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.6746283, 0.1067790, -1.6423986, 0.1199467, -1.5229564, 1.4769197
1: -17.9866734, -15.6743984, -17.9636612, -15.6950607, -1.5231647, 1.5247221
2: -6.5321455, -4.4770374, -6.5760260, -4.4971857, -1.4679480, 1.4986777
3: -13.9636040, -12.1108341, -14.0075321, -12.1495895, -1.2546649, 1.3262749
4: -5.5708370, -3.6857071, -5.5071721, -3.7193534, -1.7852221, 1.7819209
5: -7.0691161, -5.6089149, -7.0530348, -5.6135206, -1.0349765, 1.0269141
6: 8.2586994, 10.0030251, 8.3335619, 10.0840874, -1.4210472, 1.2318192
7: -13.9909163, -12.1240339, -14.0273771, -12.1290169, -1.1299167, 1.1916304
8: -6.0812349, -4.6071749, -6.0655575, -4.6518707, -0.8541272, 0.8895857
9: -10.8561506, -8.5261240, -10.8091850, -8.5342703, -1.8220325, 1.8162079

Time for backsubstitution: 5.49 seconds
Binary search (step 1): status=Status.UNKNOWN, k_low=3, k_high=6, k_mid=4, eps_mid=0.0156250, abs_max=1.3650121688842773
rel_dist={6: [-0.846481615748317, 0.8464836721348625]}

## Binary search (step 2) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1

No IS candidates found

### IS candidates at layer 3
type: A, layer: 3, pos: 2130
type: B, layer: 3, pos: 2130
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.31 seconds

### Candidate
type: A, layer: 3, pos: 2130

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7019103, upper bound: 0.7061180
time: 6.07 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7061178, upper bound: 0.7061181
time: 4.53 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 10.93 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 10.93
Output dim: 6, lower bound: -0.7019103, upper bound: 0.7061180
IS_A2, status: Status.UNKNOWN, split count: 1, time: 10.93
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

### IS candidates at layer 3
type: B, layer: 3, pos: 2130
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of IS_A1_B1

### Relational analysis result of IS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7001077, upper bound: 0.7001073
time: 4.77 seconds

## Relational analysis of IS_A1_B2

### Relational analysis result of IS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7001077, upper bound: 0.7061176
time: 4.71 seconds

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

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2130
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A2_A1

### Relational analysis result of IS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6990502, upper bound: 0.6950697
time: 5.23 seconds

## Relational analysis of IS_A2_A2

### Relational analysis result of IS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7001016, upper bound: 0.7001020
time: 4.17 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 15.02 seconds
IS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.02
Output dim: 6, lower bound: -0.7001077, upper bound: 0.7001073
IS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.02
Output dim: 6, lower bound: -0.7001077, upper bound: 0.7061176
IS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 15.02
Output dim: 6, lower bound: -0.6990502, upper bound: 0.6950697
IS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 15.02
Output dim: 6, lower bound: -0.7001016, upper bound: 0.7001020

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

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of IS_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6954817, upper bound: 0.6890772
time: 5.84 seconds

## Relational analysis of IS_A1_B1_A2

### Relational analysis result of IS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6963859, upper bound: 0.6941957
time: 3.83 seconds

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

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of IS_A1_B2_B1

### Relational analysis result of IS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6904859, upper bound: 0.6990506
time: 5.86 seconds

## Relational analysis of IS_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6963857, upper bound: 0.7001017
time: 4.35 seconds

## BFS IS instance: IS_A2_A1

### Backsubstitution after applying IS history:
0: -1.6410147, 0.1308787, -1.6476580, 0.1123592, -1.4278412, 1.4619675
1: -17.9600925, -15.6518440, -17.9592533, -15.6455917, -1.4575438, 1.4201195
2: -6.5795999, -4.5005484, -6.5322666, -4.4806995, -1.4271798, 1.3497438
3: -14.0056801, -12.1403723, -13.9775743, -12.1224070, -1.2577915, 1.2160864
4: -5.5723267, -3.7033584, -5.6068120, -3.7179585, -1.7192888, 1.7783308
5: -7.0431242, -5.6016593, -7.0480504, -5.6003075, -0.9780059, 0.9425361
6: 8.3133726, 10.0787506, 8.2846165, 10.0280476, -1.2031536, 1.2922807
7: -14.0346870, -12.1278963, -14.0092344, -12.1267395, -1.0712011, 1.0369995
8: -6.0788412, -4.6320820, -6.0965710, -4.6512518, -0.7794085, 0.7966211
9: -10.8304892, -8.5228643, -10.8328581, -8.5145502, -1.7157488, 1.7388277

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2130
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of IS_A2_A1_B1

### Relational analysis result of IS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6990501, upper bound: 0.6890766
time: 4.41 seconds

## Relational analysis of IS_A2_A1_B2

### Relational analysis result of IS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6990503, upper bound: 0.6890764
time: 7.55 seconds

## BFS IS instance: IS_A2_A2

### Backsubstitution after applying IS history:
0: -1.6419480, 0.1262433, -1.6486210, 0.1123596, -1.4301910, 1.4659090
1: -17.9519997, -15.6275787, -17.9592533, -15.6322289, -1.4708285, 1.4398313
2: -6.5837455, -4.5005035, -6.5343742, -4.4806485, -1.4319978, 1.3529425
3: -14.0151215, -12.1173935, -13.9775791, -12.1113901, -1.2724996, 1.2183881
4: -5.6089945, -3.6892047, -5.6245079, -3.7166955, -1.7226906, 1.8014803
5: -7.0418663, -5.5878100, -7.0490379, -5.5919304, -1.0065522, 0.9362124
6: 8.3123579, 10.1027765, 8.2846060, 10.0419779, -1.2344770, 1.2879119
7: -14.0354652, -12.1244431, -14.0096302, -12.1248722, -1.0726218, 1.0404451
8: -6.0946827, -4.6322107, -6.1047907, -4.6512513, -0.7841280, 0.8086472
9: -10.8474274, -8.5267429, -10.8408203, -8.5145473, -1.7331152, 1.7498631

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of IS_A2_A2_B1

### Relational analysis result of IS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7001016, upper bound: 0.6941965
time: 3.78 seconds

## Relational analysis of IS_A2_A2_B2

### Relational analysis result of IS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7001018, upper bound: 0.6941944
time: 4.27 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 13.68 seconds
IS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 13.68
Output dim: 6, lower bound: -0.6954817, upper bound: 0.6890772
IS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 13.68
Output dim: 6, lower bound: -0.6963859, upper bound: 0.6941957
IS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 13.68
Output dim: 6, lower bound: -0.6904859, upper bound: 0.6990506
IS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 13.68
Output dim: 6, lower bound: -0.6963857, upper bound: 0.7001017
IS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 13.68
Output dim: 6, lower bound: -0.6990501, upper bound: 0.6890766
IS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 13.68
Output dim: 6, lower bound: -0.6990503, upper bound: 0.6890764
IS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 13.68
Output dim: 6, lower bound: -0.7001016, upper bound: 0.6941965
IS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 13.68
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

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_A1_B1_A1_B1

### Relational analysis result of IS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6987912, upper bound: 0.6903251
time: 5.91 seconds

## Relational analysis of IS_A1_B1_A1_B2

### Relational analysis result of IS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6948322, upper bound: 0.6903226
time: 4.82 seconds

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

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_A1_B1_A2_B1

### Relational analysis result of IS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.7000582, upper bound: 0.6960837
time: 4.42 seconds

## Relational analysis of IS_A1_B1_A2_B2

### Relational analysis result of IS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6960817, upper bound: 0.6960812
time: 5.14 seconds

## BFS IS instance: IS_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.6475470, 0.1124224, -1.6410147, 0.1308787, -1.4528823, 1.4290123
1: -17.9605503, -15.6455193, -17.9600925, -15.6518440, -1.4120073, 1.4197371
2: -6.5322599, -4.4836016, -6.5795999, -4.5005484, -1.3589659, 1.4107108
3: -13.9777946, -12.1284962, -14.0056801, -12.1403723, -1.1808028, 1.2177796
4: -5.6066494, -3.7179618, -5.5723267, -3.7033584, -1.7682829, 1.7219563
5: -7.0509210, -5.6002588, -7.0431242, -5.6016593, -0.9464414, 0.9353229
6: 8.2744913, 10.0280552, 8.3133726, 10.0787506, -1.3129230, 1.1967862
7: -14.0092344, -12.1322794, -14.0346870, -12.1278963, -1.0367131, 1.0728848
8: -6.0983963, -4.6512451, -6.0788412, -4.6320820, -0.8050568, 0.7738086
9: -10.8329639, -8.5088806, -10.8304892, -8.5228643, -1.7149339, 1.7261181

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_A1_B2_B1_B1

### Relational analysis result of IS_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6846337, upper bound: 0.6889170
time: 7.45 seconds

## Relational analysis of IS_A1_B2_B1_B2

### Relational analysis result of IS_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6805510, upper bound: 0.6889170
time: 4.92 seconds

## BFS IS instance: IS_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.6485668, 0.1124225, -1.6419480, 0.1262433, -1.4568000, 1.4313626
1: -17.9605503, -15.6321545, -17.9519997, -15.6275787, -1.4303164, 1.4312642
2: -6.5343671, -4.4835496, -6.5837455, -4.5005035, -1.3621650, 1.4155278
3: -13.9778004, -12.1174812, -14.0151215, -12.1173935, -1.1818814, 1.2324891
4: -5.6234894, -3.7167022, -5.6089945, -3.6892047, -1.7914324, 1.7253551
5: -7.0519099, -5.5918832, -7.0418663, -5.5878100, -0.9401283, 0.9623913
6: 8.2744808, 10.0419827, 8.3123579, 10.1027765, -1.3077216, 1.2260985
7: -14.0096331, -12.1304121, -14.0354652, -12.1244431, -1.0401583, 1.0742552
8: -6.1066675, -4.6512427, -6.0946827, -4.6322107, -0.8170568, 0.7784804
9: -10.8409233, -8.5088787, -10.8474274, -8.5267429, -1.7259717, 1.7434859

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_A1_B2_B2_A1

### Relational analysis result of IS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6863576, upper bound: 0.6939297
time: 6.66 seconds

## Relational analysis of IS_A1_B2_B2_A2

### Relational analysis result of IS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6863576, upper bound: 0.6898445
time: 6.51 seconds

## BFS IS instance: IS_A2_A1_B1

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

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_A2_A1_B1_A1

### Relational analysis result of IS_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6889168, upper bound: 0.6831186
time: 7.76 seconds

## Relational analysis of IS_A2_A1_B1_A2

### Relational analysis result of IS_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6889168, upper bound: 0.6790432
time: 4.49 seconds

## BFS IS instance: IS_A2_A1_B2

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

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_A2_A1_B2_B1

### Relational analysis result of IS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6929925, upper bound: 0.6790415
time: 4.54 seconds

## Relational analysis of IS_A2_A1_B2_B2

### Relational analysis result of IS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6889169, upper bound: 0.6790412
time: 4.62 seconds

## BFS IS instance: IS_A2_A2_B1

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

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 1695

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_A2_A2_B1_B1

### Relational analysis result of IS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6939293, upper bound: 0.6840807
time: 7.57 seconds

## Relational analysis of IS_A2_A2_B1_B2

### Relational analysis result of IS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6898440, upper bound: 0.6840827
time: 4.76 seconds

## BFS IS instance: IS_A2_A2_B2

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

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_A2_A2_B2_B1

### Relational analysis result of IS_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6939295, upper bound: 0.6840805
time: 4.57 seconds

## Relational analysis of IS_A2_A2_B2_B2

### Relational analysis result of IS_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6898442, upper bound: 0.6840807
time: 5.38 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 15.60 seconds
IS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6987912, upper bound: 0.6903251
IS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6948322, upper bound: 0.6903226
IS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.7000582, upper bound: 0.6960837
IS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6960817, upper bound: 0.6960812
IS_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6846337, upper bound: 0.6889170
IS_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6805510, upper bound: 0.6889170
IS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6863576, upper bound: 0.6939297
IS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6863576, upper bound: 0.6898445
IS_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6889168, upper bound: 0.6831186
IS_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6889168, upper bound: 0.6790432
IS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6929925, upper bound: 0.6790415
IS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6889169, upper bound: 0.6790412
IS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6939293, upper bound: 0.6840807
IS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6898440, upper bound: 0.6840827
IS_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6939295, upper bound: 0.6840805
IS_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 15.60
Output dim: 6, lower bound: -0.6898442, upper bound: 0.6840807

## BFS IS instance: IS_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -1.6439545, 0.1168046, -1.6464876, 0.1085095, -1.4243584, 1.4339700
1: -17.9686413, -15.6656656, -17.9605465, -15.6579742, -1.4102826, 1.3931036
2: -6.5291958, -4.4839096, -6.5319557, -4.4857197, -1.3331981, 1.3379235
3: -13.9680357, -12.1420059, -13.9729252, -12.1286926, -1.1766071, 1.1681194
4: -5.5849810, -3.7301335, -5.5947914, -3.7192302, -1.7176285, 1.7165093
5: -7.0526710, -5.6109595, -7.0495882, -5.6060042, -0.9334488, 0.9343933
6: 8.2777262, 10.0088310, 8.2793732, 10.0218639, -1.2207098, 1.2191625
7: -14.0085106, -12.1355438, -14.0066519, -12.1330328, -1.0453100, 1.0378544
8: -6.0849266, -4.6511188, -6.0788097, -4.6513109, -0.7901428, 0.7814069
9: -10.8194141, -8.5055552, -10.8288984, -8.5175924, -1.7067528, 1.7286239

Time for backsubstitution: 5.44 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A1_B1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6945454, upper bound: 0.6824777
time: 9.93 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6905822, upper bound: 0.6825131
time: 5.03 seconds

## BFS IS instance: IS_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -1.6430159, 0.1149706, -1.6745574, 0.1100093, -1.4219966, 1.4614449
1: -17.9686317, -15.6751184, -17.9867172, -15.6662722, -1.4070172, 1.4122357
2: -6.5289669, -4.4851418, -6.5327020, -4.4762955, -1.3345594, 1.3504934
3: -13.9612103, -12.1421270, -13.9640236, -12.1104002, -1.1914101, 1.1683488
4: -5.5742970, -3.7309170, -5.5868912, -3.6855369, -1.7458143, 1.7073088
5: -7.0518813, -5.6144996, -7.0696073, -5.6083837, -0.9429991, 0.9575381
6: 8.2806740, 9.9971390, 8.2517681, 9.9991693, -1.2113295, 1.2619333
7: -14.0008898, -12.1361017, -13.9912119, -12.1244307, -1.0618198, 1.0353422
8: -6.0760975, -4.6512375, -6.0803165, -4.5999537, -0.8445778, 0.7920074
9: -10.8166895, -8.5122643, -10.8602858, -8.5257893, -1.7318010, 1.7597737

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A1_B2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6905815, upper bound: 0.6824795
time: 5.26 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6871012, upper bound: 0.6825130
time: 5.22 seconds

## BFS IS instance: IS_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -1.6449816, 0.1121672, -1.6475053, 0.1085109, -1.4266467, 1.4378805
1: -17.9605484, -15.6415825, -17.9605465, -15.6446657, -1.4232750, 1.4113503
2: -6.5333147, -4.4838614, -6.5340652, -4.4856668, -1.3381004, 1.3411131
3: -13.9774857, -12.1190214, -13.9729300, -12.1176720, -1.1909142, 1.1692023
4: -5.6196842, -3.7173262, -5.6116300, -3.7179687, -1.7209921, 1.7397032
5: -7.0514002, -5.5970812, -7.0505810, -5.5976262, -0.9605591, 0.9281183
6: 8.2748299, 10.0348892, 8.2793627, 10.0357857, -1.2516809, 1.2140160
7: -14.0092916, -12.1320305, -14.0070477, -12.1311569, -1.0466709, 1.0411317
8: -6.1008611, -4.6512489, -6.0870943, -4.6513085, -0.7948647, 0.7933681
9: -10.8358974, -8.5094261, -10.8368721, -8.5175896, -1.7233257, 1.7396293

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A2_B1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6957444, upper bound: 0.6882269
time: 4.83 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6918715, upper bound: 0.6883523
time: 4.50 seconds

## BFS IS instance: IS_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -1.6440468, 0.1103303, -1.6755315, 0.1100089, -1.4243112, 1.4653358
1: -17.9605408, -15.6510601, -17.9867191, -15.6530838, -1.4201956, 1.4305277
2: -6.5330739, -4.4850922, -6.5347900, -4.4762454, -1.3394661, 1.3536434
3: -13.9706602, -12.1191416, -13.9640293, -12.0993671, -1.2057133, 1.1694713
4: -5.6090021, -3.7181413, -5.6037297, -3.6842287, -1.7490854, 1.7305369
5: -7.0505719, -5.6006160, -7.0705628, -5.5999780, -0.9701157, 0.9513404
6: 8.2777662, 10.0232029, 8.2517595, 10.0131016, -1.2422867, 1.2569251
7: -14.0016708, -12.1325760, -13.9916067, -12.1225386, -1.0631189, 1.0386090
8: -6.0919600, -4.6513681, -6.0886250, -4.5999541, -0.8493199, 0.8040311
9: -10.8331804, -8.5161343, -10.8682222, -8.5257845, -1.7483749, 1.7704945

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B1_A2_B2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6916296, upper bound: 0.6882293
time: 4.92 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6883507, upper bound: 0.6883528
time: 4.14 seconds

## BFS IS instance: IS_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -1.6474741, 0.1121693, -1.6399505, 0.1269582, -1.4483204, 1.4272523
1: -17.9605484, -15.6463070, -17.9600868, -15.6643028, -1.4005976, 1.4189677
2: -6.5322385, -4.4837427, -6.5793109, -4.5026445, -1.3518434, 1.4052944
3: -13.9774885, -12.1285095, -14.0008049, -12.1405697, -1.1801867, 1.2117047
4: -5.6059041, -3.7180486, -5.5604362, -3.7045887, -1.7661448, 1.7082629
5: -7.0508318, -5.6006207, -7.0418634, -5.6074033, -0.9368124, 0.9308780
6: 8.2748299, 10.0275135, 8.3181076, 10.0725622, -1.2972248, 1.1869452
7: -14.0090742, -12.1323299, -14.0321026, -12.1286354, -1.0343487, 1.0656316
8: -6.0971498, -4.6512485, -6.0592737, -4.6321430, -0.8035834, 0.7528312
9: -10.8327026, -8.5094280, -10.8263445, -8.5315514, -1.6970844, 1.7129288

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6802583, upper bound: 0.6808681
time: 5.39 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6763526, upper bound: 0.6809986
time: 5.25 seconds

## BFS IS instance: IS_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -1.6465410, 0.1103321, -1.6677475, 0.1284892, -1.4460292, 1.4543376
1: -17.9605408, -15.6557608, -17.9862614, -15.6726294, -1.3969650, 1.4381409
2: -6.5320001, -4.4849720, -6.5800805, -4.4932203, -1.3531590, 1.4178748
3: -13.9706640, -12.1286306, -13.9918699, -12.1222744, -1.1949372, 1.2121038
4: -5.5952196, -3.7188666, -5.5523729, -3.6708283, -1.7942867, 1.6990452
5: -7.0500002, -5.6041584, -7.0620794, -5.6097002, -0.9465456, 0.9542396
6: 8.2777681, 10.0158253, 8.2910995, 10.0498381, -1.2878330, 1.2297373
7: -14.0014505, -12.1328821, -14.0166693, -12.1201248, -1.0512474, 1.0631268
8: -6.0882611, -4.6513686, -6.0609593, -4.5807586, -0.8580687, 0.7633830
9: -10.8300009, -8.5161352, -10.8575659, -8.5396843, -1.7221594, 1.7439985

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6761297, upper bound: 0.6808683
time: 4.98 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6726408, upper bound: 0.6809991
time: 5.52 seconds

## BFS IS instance: IS_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6475053, 0.1085109, -1.6418728, 0.1259893, -1.4550705, 1.4267931
1: -17.9605465, -15.6446657, -17.9519997, -15.6283646, -1.4295425, 1.4198060
2: -6.5340652, -4.4856668, -6.5837250, -4.5006399, -1.3567281, 1.4083638
3: -13.9729300, -12.1176720, -14.0148163, -12.1174068, -1.1759119, 1.2318749
4: -5.6116300, -3.7179687, -5.6082454, -3.6892915, -1.7777634, 1.7232070
5: -7.0505810, -5.5976262, -7.0417786, -5.5881705, -0.9356432, 0.9528999
6: 8.2793627, 10.0357857, 8.3126898, 10.1022282, -1.2975621, 1.2103682
7: -14.0070477, -12.1311569, -14.0353031, -12.1244898, -1.0329084, 1.0718205
8: -6.0870943, -4.6513085, -6.0934443, -4.6322145, -0.7960150, 0.7770296
9: -10.8368721, -8.5175896, -10.8471613, -8.5272875, -1.7128539, 1.7256255

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_B2_A1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6782956, upper bound: 0.6895319
time: 7.34 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6784391, upper bound: 0.6856588
time: 5.91 seconds

## BFS IS instance: IS_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6755315, 0.1100089, -1.6409276, 0.1241539, -1.4825273, 1.4244695
1: -17.9867191, -15.6530838, -17.9519882, -15.6378746, -1.4486547, 1.4167295
2: -6.5347900, -4.4762454, -6.5834851, -4.5018606, -1.3692708, 1.4097252
3: -13.9640293, -12.0993671, -14.0079832, -12.1175261, -1.1761789, 1.2466197
4: -5.6037297, -3.6842287, -5.5975342, -3.6901114, -1.7686100, 1.7512918
5: -7.0705628, -5.5999780, -7.0409479, -5.5917063, -0.9588470, 0.9624815
6: 8.2517595, 10.0131016, 8.3155327, 10.0905380, -1.3404679, 1.2011452
7: -13.9916067, -12.1225386, -14.0276794, -12.1250219, -1.0304275, 1.0882695
8: -6.0886250, -4.5999541, -6.0844588, -4.6323295, -0.8066857, 0.8316664
9: -10.8682222, -8.5257845, -10.8444090, -8.5339746, -1.7437201, 1.7506046

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2148

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A1_B2_B2_A2_B1

### Relational analysis result of IS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6782956, upper bound: 0.6854030
time: 6.16 seconds

## Relational analysis of IS_A1_B2_B2_A2_B2

### Relational analysis result of IS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6784391, upper bound: 0.6819416
time: 6.26 seconds

## BFS IS instance: IS_A2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6399505, 0.1269582, -1.6474741, 0.1121693, -1.4272523, 1.4483204
1: -17.9600868, -15.6643028, -17.9605484, -15.6463070, -1.4189677, 1.4005973
2: -6.5793109, -4.5026445, -6.5322385, -4.4837427, -1.4052944, 1.3518434
3: -14.0008049, -12.1405697, -13.9774885, -12.1285095, -1.2117047, 1.1801867
4: -5.5604362, -3.7045887, -5.6059041, -3.7180486, -1.7082624, 1.7661448
5: -7.0418634, -5.6074033, -7.0508318, -5.6006207, -0.9308779, 0.9368124
6: 8.3181076, 10.0725622, 8.2748299, 10.0275135, -1.1869454, 1.2972245
7: -14.0321026, -12.1286354, -14.0090742, -12.1323299, -1.0656316, 1.0343487
8: -6.0592737, -4.6321430, -6.0971498, -4.6512485, -0.7528312, 0.8035834
9: -10.8263445, -8.5315514, -10.8327026, -8.5094280, -1.7129288, 1.6970844

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_A1_B1_A1_B1

### Relational analysis result of IS_A2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6808680, upper bound: 0.6802604
time: 6.43 seconds

## Relational analysis of IS_A2_A1_B1_A1_B2

### Relational analysis result of IS_A2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6809988, upper bound: 0.6763548
time: 4.70 seconds

## BFS IS instance: IS_A2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6677475, 0.1284892, -1.6465410, 0.1103321, -1.4543376, 1.4460287
1: -17.9862614, -15.6726294, -17.9605408, -15.6557608, -1.4381409, 1.3969650
2: -6.5800805, -4.4932203, -6.5320001, -4.4849720, -1.4178748, 1.3531590
3: -13.9918699, -12.1222744, -13.9706640, -12.1286306, -1.2121038, 1.1949372
4: -5.5523729, -3.6708283, -5.5952196, -3.7188666, -1.6990452, 1.7942863
5: -7.0620794, -5.6097002, -7.0500002, -5.6041584, -0.9542396, 0.9465456
6: 8.2910995, 10.0498381, 8.2777681, 10.0158253, -1.2297373, 1.2878332
7: -14.0166693, -12.1201248, -14.0014505, -12.1328821, -1.0631268, 1.0512471
8: -6.0609593, -4.5807586, -6.0882611, -4.6513686, -0.7633829, 0.8580687
9: -10.8575659, -8.5396843, -10.8300009, -8.5161352, -1.7439981, 1.7221594

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of IS_A2_A1_B1_A2_B1

### Relational analysis result of IS_A2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6808680, upper bound: 0.6761300
time: 6.96 seconds

## Relational analysis of IS_A2_A1_B1_A2_B2

### Relational analysis result of IS_A2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6809988, upper bound: 0.6726409
time: 4.48 seconds

## BFS IS instance: IS_A2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -1.6409403, 0.1306250, -1.6434035, 0.1223214, -1.4470654, 1.4567065
1: -17.9600925, -15.6526318, -17.9519939, -15.6449585, -1.4486790, 1.4319043
2: -6.5795832, -4.5006862, -6.5823689, -4.5024738, -1.3454809, 1.3502078
3: -14.0053749, -12.1403904, -14.0102510, -12.1270752, -1.1975565, 1.1885114
4: -5.5715761, -3.7034416, -5.5820374, -3.6911969, -1.7166901, 1.7155738
5: -7.0430384, -5.6020203, -7.0399675, -5.5971160, -0.9687622, 0.9712532
6: 8.3137093, 10.0782194, 8.3170681, 10.0905628, -1.2052183, 1.2040207
7: -14.0345221, -12.1279430, -14.0326643, -12.1254559, -1.0609658, 1.0535321
8: -6.0776005, -4.6320844, -6.0714273, -4.6322746, -0.7767065, 0.7679455
9: -10.8302221, -8.5234108, -10.8399744, -8.5354214, -1.7034378, 1.7253170

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_A1_B2_B1_A1

### Relational analysis result of IS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6885980, upper bound: 0.6709813
time: 4.81 seconds

## Relational analysis of IS_A2_A1_B2_B1_A2

### Relational analysis result of IS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6847036, upper bound: 0.6711258
time: 6.47 seconds

## BFS IS instance: IS_A2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -1.6399957, 0.1287923, -1.6711860, 0.1238356, -1.4447536, 1.4838438
1: -17.9600830, -15.6620865, -17.9781704, -15.6532869, -1.4455671, 1.4511395
2: -6.5793543, -4.5019031, -6.5830994, -4.4930544, -1.3466539, 1.3627520
3: -13.9985428, -12.1405067, -14.0013142, -12.1087828, -1.2124887, 1.1888118
4: -5.5608654, -3.7042406, -5.5739737, -3.6575155, -1.7448740, 1.7064147
5: -7.0422497, -5.6055579, -7.0599494, -5.5993743, -0.9783564, 0.9943411
6: 8.3165627, 10.0665169, 8.2899609, 10.0678444, -1.1957366, 1.2462659
7: -14.0269012, -12.1284914, -14.0172300, -12.1169329, -1.0778112, 1.0510752
8: -6.0686874, -4.6322002, -6.0730705, -4.5808883, -0.8313107, 0.7782683
9: -10.8274670, -8.5301027, -10.8711472, -8.5435457, -1.7284851, 1.7563763

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_A1_B2_B2_A1

### Relational analysis result of IS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6844863, upper bound: 0.6709814
time: 4.85 seconds

## Relational analysis of IS_A2_A1_B2_B2_A2

### Relational analysis result of IS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6809988, upper bound: 0.6711247
time: 5.52 seconds

## BFS IS instance: IS_A2_A2_B1_B1

### Backsubstitution after applying IS history:
0: -1.6418728, 0.1259893, -1.6475053, 0.1085109, -1.4267931, 1.4550705
1: -17.9519997, -15.6283646, -17.9605465, -15.6446657, -1.4198060, 1.4295425
2: -6.5837250, -4.5006399, -6.5340652, -4.4856668, -1.4083638, 1.3567281
3: -14.0148163, -12.1174068, -13.9729300, -12.1176720, -1.2318749, 1.1759119
4: -5.6082454, -3.6892915, -5.6116300, -3.7179687, -1.7232075, 1.7777634
5: -7.0417786, -5.5881705, -7.0505810, -5.5976262, -0.9528999, 0.9356434
6: 8.3126898, 10.1022282, 8.2793627, 10.0357857, -1.2103682, 1.2975621
7: -14.0353031, -12.1244898, -14.0070477, -12.1311569, -1.0718203, 1.0329080
8: -6.0934443, -4.6322145, -6.0870943, -4.6513085, -0.7770295, 0.7960150
9: -10.8471613, -8.5272875, -10.8368721, -8.5175896, -1.7256250, 1.7128534

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_A2_B1_B1_A1

### Relational analysis result of IS_A2_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6895296, upper bound: 0.6782962
time: 5.58 seconds

## Relational analysis of IS_A2_A2_B1_B1_A2

### Relational analysis result of IS_A2_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6856565, upper bound: 0.6784394
time: 6.75 seconds

## BFS IS instance: IS_A2_A2_B1_B2

### Backsubstitution after applying IS history:
0: -1.6409276, 0.1241539, -1.6755315, 0.1100089, -1.4244690, 1.4825273
1: -17.9519882, -15.6378746, -17.9867191, -15.6530838, -1.4167295, 1.4486547
2: -6.5834851, -4.5018606, -6.5347900, -4.4762454, -1.4097252, 1.3692708
3: -14.0079832, -12.1175261, -13.9640293, -12.0993671, -1.2466197, 1.1761789
4: -5.5975342, -3.6901114, -5.6037297, -3.6842287, -1.7512913, 1.7686095
5: -7.0409479, -5.5917063, -7.0705628, -5.5999780, -0.9624815, 0.9588470
6: 8.3155327, 10.0905380, 8.2517595, 10.0131016, -1.2011447, 1.3404679
7: -14.0276794, -12.1250219, -13.9916067, -12.1225386, -1.0882692, 1.0304277
8: -6.0844588, -4.6323295, -6.0886250, -4.5999541, -0.8316662, 0.8066857
9: -10.8444090, -8.5339746, -10.8682222, -8.5257845, -1.7506046, 1.7437205

Time for backsubstitution: 5.45 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_A2_B1_B2_A1

### Relational analysis result of IS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6854029, upper bound: 0.6782957
time: 5.76 seconds

## Relational analysis of IS_A2_A2_B1_B2_A2

### Relational analysis result of IS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6819408, upper bound: 0.6784395
time: 4.45 seconds

## BFS IS instance: IS_A2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -1.6418728, 0.1259893, -1.6443690, 0.1223218, -1.4494171, 1.4606485
1: -17.9519997, -15.6283646, -17.9519958, -15.6314230, -1.4622922, 1.4492555
2: -6.5837250, -4.5006399, -6.5844784, -4.5024257, -1.3502903, 1.3533096
3: -14.0148163, -12.1174068, -14.0102568, -12.1160583, -1.2104697, 1.1904349
4: -5.6082454, -3.6892915, -5.5997314, -3.6899471, -1.7200594, 1.7387862
5: -7.0417786, -5.5881705, -7.0409584, -5.5887146, -0.9973531, 0.9632580
6: 8.3126898, 10.1022282, 8.3170605, 10.1036005, -1.2365727, 1.1986001
7: -14.0353031, -12.1244898, -14.0330601, -12.1236067, -1.0625994, 1.0569496
8: -6.0934443, -4.6322145, -6.0796585, -4.6322732, -0.7814267, 0.7799792
9: -10.8471613, -8.5272875, -10.8481770, -8.5354223, -1.7200079, 1.7363238

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_A2_B2_B1_A1

### Relational analysis result of IS_A2_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6895298, upper bound: 0.6760332
time: 6.09 seconds

## Relational analysis of IS_A2_A2_B2_B1_A2

### Relational analysis result of IS_A2_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6856565, upper bound: 0.6761617
time: 4.51 seconds

## BFS IS instance: IS_A2_A2_B2_B2

### Backsubstitution after applying IS history:
0: -1.6409276, 0.1241539, -1.6721075, 0.1238346, -1.4471245, 1.4877615
1: -17.9519882, -15.6378746, -17.9781666, -15.6399431, -1.4593658, 1.4685121
2: -6.5834851, -4.5018606, -6.5852041, -4.4930029, -1.3514709, 1.3658161
3: -14.0079832, -12.1175261, -14.0013227, -12.0977516, -1.2253933, 1.1907897
4: -5.5975342, -3.6901114, -5.5916662, -3.6562212, -1.7481508, 1.7296576
5: -7.0409479, -5.5917063, -7.0609035, -5.5909467, -1.0069537, 0.9864230
6: 8.3155327, 10.0905380, 8.2899532, 10.0808945, -1.2270753, 1.2409794
7: -14.0276794, -12.1250219, -14.0176268, -12.1150684, -1.0794005, 1.0544810
8: -6.0844588, -4.6323295, -6.0813293, -4.5808868, -0.8360255, 0.7903771
9: -10.8444090, -8.5339746, -10.8793106, -8.5435448, -1.7450533, 1.7670999

Time for backsubstitution: 5.46 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of IS_A2_A2_B2_B2_A1

### Relational analysis result of IS_A2_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6854029, upper bound: 0.6760345
time: 4.51 seconds

## Relational analysis of IS_A2_A2_B2_B2_A2

### Relational analysis result of IS_A2_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6819408, upper bound: 0.6761625
time: 4.44 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 14.58 seconds
IS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6945454, upper bound: 0.6824777
IS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6905822, upper bound: 0.6825131
IS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6905815, upper bound: 0.6824795
IS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6871012, upper bound: 0.6825130
IS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6957444, upper bound: 0.6882269
IS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6918715, upper bound: 0.6883523
IS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6916296, upper bound: 0.6882293
IS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6883507, upper bound: 0.6883528
IS_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6802583, upper bound: 0.6808681
IS_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6763526, upper bound: 0.6809986
IS_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6761297, upper bound: 0.6808683
IS_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6726408, upper bound: 0.6809991
IS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6782956, upper bound: 0.6895319
IS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6784391, upper bound: 0.6856588
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6782956, upper bound: 0.6854030
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6784391, upper bound: 0.6819416
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6808680, upper bound: 0.6802604
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6809988, upper bound: 0.6763548
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6808680, upper bound: 0.6761300
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6809988, upper bound: 0.6726409
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6885980, upper bound: 0.6709813
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6847036, upper bound: 0.6711258
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6844863, upper bound: 0.6709814
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6809988, upper bound: 0.6711247
IS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6895296, upper bound: 0.6782962
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6856565, upper bound: 0.6784394
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6854029, upper bound: 0.6782957
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6819408, upper bound: 0.6784395
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6895298, upper bound: 0.6760332
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6856565, upper bound: 0.6761617
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6854029, upper bound: 0.6760345
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 14.58
Output dim: 6, lower bound: -0.6819408, upper bound: 0.6761625

## BFS IS instance: IS_A1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6463423, 0.1075161, -1.6456689, 0.1042511, -1.4189916, 1.4175253
1: -17.9722176, -15.7010508, -17.9604855, -15.6739120, -1.3922813, 1.3557639
2: -6.5258055, -4.4795008, -6.5304308, -4.4867215, -1.3252983, 1.3364444
3: -13.9752216, -12.1511173, -13.9723473, -12.1328011, -1.1763921, 1.1559868
4: -5.5282907, -3.7454975, -5.5683002, -3.7201054, -1.6644812, 1.6628585
5: -7.0631680, -5.6199312, -7.0485439, -5.6100440, -0.9357560, 0.9210231
6: 8.2961464, 10.0236149, 8.2878704, 10.0218039, -1.1915660, 1.2060156
7: -14.0074072, -12.1362782, -14.0061321, -12.1333609, -1.0437706, 1.0361741
8: -6.0794964, -4.6708179, -6.0767193, -4.6602139, -0.7714839, 0.7563397
9: -10.8003874, -8.5115318, -10.8201885, -8.5180836, -1.6867404, 1.7149959

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_A1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6945454, upper bound: 0.6824765
time: 8.21 seconds

## Relational analysis of IS_A1_B1_A1_B1_A1_A2

### Relational analysis result of IS_A1_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6945454, upper bound: 0.6824777
time: 10.81 seconds

## BFS IS instance: IS_A1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6428974, 0.1084190, -1.6460569, 0.1050799, -1.4153824, 1.4287004
1: -17.9683781, -15.6860275, -17.9604378, -15.6666050, -1.3980031, 1.3719206
2: -6.5191712, -4.4853344, -6.5279074, -4.4863272, -1.3259897, 1.3343310
3: -13.9672518, -12.1538706, -13.9725857, -12.1334820, -1.1723967, 1.1625352
4: -5.5775452, -3.7309403, -5.5913930, -3.7195945, -1.6588774, 1.6982851
5: -7.0514994, -5.6284685, -7.0490761, -5.6130686, -0.9281552, 0.9303832
6: 8.3009195, 10.0087528, 8.2895994, 10.0218277, -1.1962044, 1.2110534
7: -14.0077715, -12.1359215, -14.0063515, -12.1331930, -1.0448170, 1.0365305
8: -6.0827560, -4.6584249, -6.0779200, -4.6542711, -0.7822676, 0.7664279
9: -10.8164349, -8.5060654, -10.8276691, -8.5177994, -1.7063193, 1.7155428

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_A1_B1_A1_B1_A2_A1

### Relational analysis result of IS_A1_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6905822, upper bound: 0.6825132
time: 4.90 seconds

## Relational analysis of IS_A1_B1_A1_B1_A2_A2

### Relational analysis result of IS_A1_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6905822, upper bound: 0.6825131
time: 4.85 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6454344, 0.1055888, -1.6737387, 0.1056762, -1.4166155, 1.4447875
1: -17.9722061, -15.7103367, -17.9866600, -15.6822777, -1.3904352, 1.3758116
2: -6.5255766, -4.4807911, -6.5311770, -4.4773350, -1.3267355, 1.3489385
3: -13.9683752, -12.1512403, -13.9634571, -12.1145220, -1.1898909, 1.1562304
4: -5.5177240, -3.7461791, -5.5599418, -3.6862555, -1.6926718, 1.6538363
5: -7.0624485, -5.6234670, -7.0685916, -5.6124306, -0.9450858, 0.9436691
6: 8.2984962, 10.0118198, 8.2609558, 9.9991083, -1.1835737, 1.2498207
7: -13.9997864, -12.1367989, -13.9906683, -12.1247292, -1.0603881, 1.0337698
8: -6.0709591, -4.6709428, -6.0783610, -4.6096296, -0.8251781, 0.7671518
9: -10.7975492, -8.5181713, -10.8512926, -8.5262403, -1.7118964, 1.7436748

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6775599, upper bound: 0.6631517
time: 6.24 seconds

## Relational analysis of IS_A1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A1_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6803635, upper bound: 0.6727389
time: 5.61 seconds

## BFS IS instance: IS_A1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6419244, 0.1065723, -1.6740962, 0.1065766, -1.4131308, 1.4563351
1: -17.9683685, -15.6953144, -17.9866066, -15.6746626, -1.3949780, 1.3958485
2: -6.5189419, -4.4865861, -6.5286570, -4.4768677, -1.3274202, 1.3468413
3: -13.9604092, -12.1539907, -13.9637270, -12.1151619, -1.1871910, 1.1621079
4: -5.5671959, -3.7318423, -5.5842085, -3.6859539, -1.6886072, 1.6879320
5: -7.0506248, -5.6320014, -7.0690632, -5.6154265, -0.9375987, 0.9526637
6: 8.3043880, 9.9970579, 8.2610016, 9.9991398, -1.1871748, 1.2537198
7: -14.0001459, -12.1365242, -13.9909277, -12.1246185, -1.0613899, 1.0339558
8: -6.0736485, -4.6585460, -6.0791869, -4.6030641, -0.8366089, 0.7753670
9: -10.8137312, -8.5128326, -10.8590841, -8.5260496, -1.7320228, 1.7462778

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_A1_B1_A1_B2_A2_A1

### Relational analysis result of IS_A1_B1_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6871012, upper bound: 0.6825137
time: 7.05 seconds

## Relational analysis of IS_A1_B1_A1_B2_A2_A2

### Relational analysis result of IS_A1_B1_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6871012, upper bound: 0.6825130
time: 5.23 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -1.6474124, 0.1028848, -1.6466837, 0.1042535, -1.4213543, 1.4214430
1: -17.9641247, -15.6771049, -17.9604893, -15.6606903, -1.4053440, 1.3739026
2: -6.5299253, -4.4794493, -6.5325413, -4.4866705, -1.3302007, 1.3396168
3: -13.9846706, -12.1281300, -13.9723530, -12.1217804, -1.1906967, 1.1570706
4: -5.5629959, -3.7328317, -5.5851407, -3.7188430, -1.6678247, 1.6862035
5: -7.0619402, -5.6060333, -7.0495176, -5.6016636, -0.9627588, 0.9147915
6: 8.2932110, 10.0497036, 8.2878628, 10.0357285, -1.2225142, 1.2009706
7: -14.0081882, -12.1327677, -14.0065260, -12.1314869, -1.0451260, 1.0394394
8: -6.0954008, -4.6709461, -6.0850186, -4.6602135, -0.7763865, 0.7683353
9: -10.8168793, -8.5153980, -10.8281631, -8.5180817, -1.7033362, 1.7259960

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_A1_B1_A2_B1_A1_A1

### Relational analysis result of IS_A1_B1_A2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6957444, upper bound: 0.6881912
time: 7.35 seconds

## Relational analysis of IS_A1_B1_A2_B1_A1_A2

### Relational analysis result of IS_A1_B1_A2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6957444, upper bound: 0.6882269
time: 4.87 seconds

## BFS IS instance: IS_A1_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -1.6439261, 0.1037849, -1.6470765, 0.1050810, -1.4177051, 1.4326534
1: -17.9602852, -15.6617279, -17.9604397, -15.6532421, -1.4110141, 1.3900185
2: -6.5232878, -4.4852862, -6.5300164, -4.4862723, -1.3309283, 1.3375216
3: -13.9767036, -12.1308861, -13.9725895, -12.1224632, -1.1867018, 1.1636190
4: -5.6122503, -3.7181892, -5.6082320, -3.7183323, -1.6622925, 1.7214956
5: -7.0501857, -5.6145778, -7.0500598, -5.6046877, -0.9552233, 0.9241066
6: 8.2980022, 10.0348082, 8.2895889, 10.0357504, -1.2272105, 1.2059183
7: -14.0085487, -12.1324100, -14.0067482, -12.1313171, -1.0461760, 1.0398009
8: -6.0987067, -4.6585536, -6.0862103, -4.6542692, -0.7869881, 0.7785077
9: -10.8329239, -8.5099335, -10.8356438, -8.5177984, -1.7227807, 1.7265425

Time for backsubstitution: 5.48 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_A1_B1_A2_B1_A2_A1

### Relational analysis result of IS_A1_B1_A2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6918715, upper bound: 0.6883503
time: 4.51 seconds

## Relational analysis of IS_A1_B1_A2_B1_A2_A2

### Relational analysis result of IS_A1_B1_A2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6918715, upper bound: 0.6883523
time: 4.45 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -1.6465067, 0.1009563, -1.6747124, 0.1056777, -1.4189925, 1.4486837
1: -17.9641151, -15.6864138, -17.9866600, -15.6691437, -1.4036546, 1.3939903
2: -6.5296836, -4.4807429, -6.5332680, -4.4772825, -1.3316469, 1.3520751
3: -13.9778252, -12.1282530, -13.9634657, -12.1034889, -1.2041898, 1.1573491
4: -5.5524273, -3.7335379, -5.5767817, -3.6849475, -1.6959295, 1.6772165
5: -7.0611992, -5.6095657, -7.0695467, -5.6040235, -0.9720993, 0.9375082
6: 8.2955494, 10.0379181, 8.2609463, 10.0130434, -1.2145042, 1.2448683
7: -14.0005674, -12.1332731, -13.9910650, -12.1228380, -1.0616815, 1.0370255
8: -6.0868421, -4.6710701, -6.0866842, -4.6096287, -0.8301179, 0.7792181
9: -10.8140440, -8.5220337, -10.8592300, -8.5262413, -1.7284713, 1.7543831

Time for backsubstitution: 5.47 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_A1_B1_A2_B2_A1_A1

### Relational analysis result of IS_A1_B1_A2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6916296, upper bound: 0.6881910
time: 4.82 seconds

## Relational analysis of IS_A1_B1_A2_B2_A1_A2

### Relational analysis result of IS_A1_B1_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6916296, upper bound: 0.6882293
time: 4.97 seconds

## BFS IS instance: IS_A1_B1_A2_B2_A2

### Backsubstitution after applying IS history:
0: -1.6429570, 0.1019362, -1.6750693, 0.1065762, -1.4154677, 1.4602695
1: -17.9602737, -15.6710377, -17.9866085, -15.6614246, -1.4081349, 1.4139957
2: -6.5230508, -4.4865375, -6.5307436, -4.4768147, -1.3323717, 1.3500099
3: -13.9698601, -12.1310043, -13.9637318, -12.1041288, -1.2014894, 1.1632285
4: -5.6019001, -3.7191348, -5.6010485, -3.6846476, -1.6919308, 1.7111831
5: -7.0492702, -5.6181073, -7.0700192, -5.6070194, -0.9646664, 0.9464753
6: 8.3014584, 10.0231247, 8.2609911, 10.0130730, -1.2181659, 1.2487149
7: -14.0009260, -12.1329975, -13.9913244, -12.1227264, -1.0626838, 1.0372162
8: -6.0895300, -4.6586771, -6.0875034, -4.6030617, -0.8413446, 0.7875221
9: -10.8302202, -8.5166960, -10.8670216, -8.5260487, -1.7484636, 1.7569938

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1695
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_A1_B1_A2_B2_A2_A1

### Relational analysis result of IS_A1_B1_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6883507, upper bound: 0.6883523
time: 4.69 seconds

## Relational analysis of IS_A1_B1_A2_B2_A2_A2

### Relational analysis result of IS_A1_B1_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6883507, upper bound: 0.6883528
time: 4.49 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -1.6498952, 0.1028863, -1.6391191, 0.1227074, -1.4430137, 1.4107919
1: -17.9641266, -15.6817799, -17.9600315, -15.6803570, -1.3824003, 1.3815229
2: -6.5288515, -4.4793291, -6.5777893, -4.5036244, -1.3439798, 1.4037986
3: -13.9846745, -12.1376228, -14.0002193, -12.1446781, -1.1799645, 1.1995487
4: -5.5492144, -3.7335448, -5.5339756, -3.7054212, -1.7129946, 1.6547718
5: -7.0613475, -5.6095824, -7.0408564, -5.6114416, -0.9393346, 0.9176159
6: 8.2932100, 10.0423183, 8.3264723, 10.0725021, -1.2680557, 1.1740525
7: -14.0079699, -12.1330643, -14.0315819, -12.1289463, -1.0328355, 1.0639553
8: -6.0916767, -4.6709476, -6.0572438, -4.6410484, -0.7850641, 0.7277842
9: -10.8136816, -8.5153980, -10.8176708, -8.5320206, -1.6770959, 1.6992974

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_A1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_B2_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6802583, upper bound: 0.6808677
time: 4.41 seconds

## Relational analysis of IS_A1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_B2_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6802583, upper bound: 0.6808681
time: 5.33 seconds

## BFS IS instance: IS_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -1.6464212, 0.1037869, -1.6395146, 0.1235357, -1.4393635, 1.4217701
1: -17.9602833, -15.6665115, -17.9599800, -15.6724796, -1.3888683, 1.3976660
2: -6.5222130, -4.4851661, -6.5752668, -4.5032406, -1.3446836, 1.4016671
3: -13.9767056, -12.1403770, -14.0004559, -12.1453590, -1.1759710, 1.2060990
4: -5.5984697, -3.7189116, -5.5570450, -3.7049365, -1.7073145, 1.6900434
5: -7.0496178, -5.6181221, -7.0413651, -5.6143694, -0.9314909, 0.9268918
6: 8.2980032, 10.0274334, 8.3285809, 10.0725250, -1.2726259, 1.1790185
7: -14.0083342, -12.1327095, -14.0318022, -12.1287823, -1.0338678, 1.0643125
8: -6.0949807, -4.6585546, -6.0584092, -4.6351042, -0.7957163, 0.7373356
9: -10.8297262, -8.5099344, -10.8251324, -8.5317478, -1.6965728, 1.6998596

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of IS_A1_B2_B1_B1_A2_A1

### Relational analysis result of IS_A1_B2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6763526, upper bound: 0.6809986
time: 5.17 seconds

## Relational analysis of IS_A1_B2_B1_B1_A2_A2

### Relational analysis result of IS_A1_B2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6763526, upper bound: 0.6809986
time: 5.14 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -1.6489923, 0.1009583, -1.6669158, 0.1241618, -1.4407077, 1.4376512
1: -17.9641151, -15.6910706, -17.9862003, -15.6887608, -1.3802829, 1.4016125
2: -6.5286102, -4.4806237, -6.5785542, -4.4942408, -1.3453712, 1.4163198
3: -13.9778280, -12.1377449, -13.9912949, -12.1263981, -1.1934094, 1.1999640
4: -5.5386457, -3.7342548, -5.5255041, -3.6715136, -1.7411404, 1.6457253
5: -7.0606036, -5.6131158, -7.0610943, -5.6137457, -0.9488108, 0.9404638
6: 8.2955494, 10.0305300, 8.2998867, 10.0497770, -1.2600496, 1.2178068
7: -14.0003510, -12.1335773, -14.0161266, -12.1204138, -1.0498290, 1.0615585
8: -6.0831170, -4.6710711, -6.0591030, -4.5904341, -0.8388062, 0.7385914
9: -10.8108673, -8.5220346, -10.8486109, -8.5401192, -1.7022610, 1.7279081

Time for backsubstitution: 5.49 seconds

### IS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 2349
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 1695
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 2148

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of IS_A1_B2_B1_B2_A1_A1

### Relational analysis result of IS_A1_B2_B1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6623665, upper bound: 0.6587198
time: 6.30 seconds

## Relational analysis of IS_A1_B2_B1_B2_A1_A2

### Relational analysis result of IS_A1_B2_B1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6664660, upper bound: 0.6709246
time: 5.20 seconds

## BFS IS instance: IS_A1_B2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -1.6454515, 0.1019383, -1.6672792, 0.1250600, -1.4371810, 1.4492135
1: -17.9602737, -15.6757956, -17.9861526, -15.6807232, -1.3851447, 1.4216375
2: -6.5219746, -4.4864187, -6.5760312, -4.4937782, -1.3460584, 1.4142046
3: -13.9698629, -12.1404963, -13.9915638, -12.1270390, -1.1907115, 1.2057323
4: -5.5881171, -3.7198596, -5.5496979, -3.6712272, -1.7369995, 1.6796980
5: -7.0487103, -5.6216540, -7.0615516, -5.6167440, -0.9408801, 0.9493690
6: 8.3014584, 10.0157433, 8.3003445, 10.0498085, -1.2635906, 1.2217197
7: -14.0007086, -12.1333036, -14.0163860, -12.1203051, -1.0508518, 1.0617461
8: -6.0858107, -4.6586761, -6.0598817, -4.5838675, -0.8501053, 0.7462577
9: -10.8270473, -8.5166960, -10.8563824, -8.5399370, -1.7222910, 1.7305121

Time for backsubstitution: 5.51 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1997
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2917
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 555

## Relational analysis of IS_A1_B2_B1_B2_A2_B1

### Relational analysis result of IS_A1_B2_B1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6660722, upper bound: 0.6749682
time: 4.76 seconds

## Relational analysis of IS_A1_B2_B1_B2_A2_B2

### Relational analysis result of IS_A1_B2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6663653, upper bound: 0.6721294
time: 4.59 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -1.6466837, 0.1042535, -1.6440454, 0.1167248, -1.4386148, 1.4213042
1: -17.9604893, -15.6606903, -17.9555779, -15.6641779, -1.3916974, 1.4018383
2: -6.5325413, -4.4866705, -6.5803361, -4.4961934, -1.3552060, 1.4004641
3: -13.9723530, -12.1217804, -14.0220642, -12.1265163, -1.1637645, 1.2314897
4: -5.5851407, -3.7188430, -5.5516558, -3.7048154, -1.7242889, 1.6701236
5: -7.0495176, -5.6016636, -7.0523314, -5.5970745, -0.9225562, 0.9551427
6: 8.2878628, 10.0357285, 8.3308334, 10.1170473, -1.2845087, 1.1814656
7: -14.0065260, -12.1314869, -14.0342064, -12.1251860, -1.0312867, 1.0702908
8: -6.0850186, -4.6602135, -6.0879831, -4.6519127, -0.7709941, 0.7583572
9: -10.8281631, -8.5180817, -10.8282528, -8.5332041, -1.6992254, 1.7056975

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 577
type: B, layer: 3, pos: 577
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 718
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 1695
type: A, layer: 3, pos: 1997
type: B, layer: 3, pos: 414
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 886
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2349
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 2234
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2148
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 2917
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 615
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2305
type: A, layer: 3, pos: 1695

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_A1_B2_B2_A1_B1_B1

### Relational analysis result of IS_A1_B2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6782956, upper bound: 0.6895300
time: 9.01 seconds

## Relational analysis of IS_A1_B2_B2_A1_B1_B2

### Relational analysis result of IS_A1_B2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6782956, upper bound: 0.6895318
time: 8.98 seconds

## BFS IS instance: IS_A1_B2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -1.6470765, 0.1050810, -1.6408036, 0.1176269, -1.4498425, 1.4176598
1: -17.9604397, -15.6532421, -17.9517365, -15.6475878, -1.4077864, 1.4075103
2: -6.5300164, -4.4862723, -6.5737000, -4.5020409, -1.3531394, 1.4011922
3: -13.9725895, -12.1224632, -14.0140076, -12.1292744, -1.1703224, 1.2276187
4: -5.6082320, -3.7183323, -5.6007795, -3.6901639, -1.7595654, 1.6645608
5: -7.0500598, -5.6046877, -7.0405636, -5.6051664, -0.9318473, 0.9476074
6: 8.2895889, 10.0357504, 8.3361263, 10.1021500, -1.2894607, 1.1858921
7: -14.0067482, -12.1313171, -14.0345688, -12.1248436, -1.0316110, 1.0713377
8: -6.0862103, -4.6542692, -6.0913491, -4.6395206, -0.7811657, 0.7688735
9: -10.8356438, -8.5177984, -10.8442354, -8.5277672, -1.6997700, 1.7251368

Time for backsubstitution: 5.50 seconds

### IS candidates at layer 3
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 577
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 577
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 1732
type: A, layer: 3, pos: 1732
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 2563
type: A, layer: 3, pos: 2563
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 718
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 718
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 414
type: B, layer: 3, pos: 1997
type: B, layer: 3, pos: 1695
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1997
type: A, layer: 3, pos: 414
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2349
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 886
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 886
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2234
type: B, layer: 3, pos: 2234
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 2148
type: B, layer: 3, pos: 2461
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2461
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 2349
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2917
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2917
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 615
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 615
type: A, layer: 3, pos: 2305
type: B, layer: 3, pos: 2305
type: A, layer: 3, pos: 2148
type: A, layer: 3, pos: 1695

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of IS_A1_B2_B2_A1_B2_B1

### Relational analysis result of IS_A1_B2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6784391, upper bound: 0.6856586
time: 7.14 seconds

## Relational analysis of IS_A1_B2_B2_A1_B2_B2

### Relational analysis result of IS_A1_B2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6784391, upper bound: 0.6856586
time: 5.28 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 18.10 seconds
IS_A1_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6945454, upper bound: 0.6824765
IS_A1_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6945454, upper bound: 0.6824777
IS_A1_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6905822, upper bound: 0.6825132
IS_A1_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6905822, upper bound: 0.6825131
IS_A1_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6775599, upper bound: 0.6631517
IS_A1_B1_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6803635, upper bound: 0.6727389
IS_A1_B1_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6871012, upper bound: 0.6825137
IS_A1_B1_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6871012, upper bound: 0.6825130
IS_A1_B1_A2_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6957444, upper bound: 0.6881912
IS_A1_B1_A2_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6957444, upper bound: 0.6882269
IS_A1_B1_A2_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6918715, upper bound: 0.6883503
IS_A1_B1_A2_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6918715, upper bound: 0.6883523
IS_A1_B1_A2_B2_A1_A1, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6916296, upper bound: 0.6881910
IS_A1_B1_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6916296, upper bound: 0.6882293
IS_A1_B1_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6883507, upper bound: 0.6883523
IS_A1_B1_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6883507, upper bound: 0.6883528
IS_A1_B2_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6802583, upper bound: 0.6808677
IS_A1_B2_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6802583, upper bound: 0.6808681
IS_A1_B2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6763526, upper bound: 0.6809986
IS_A1_B2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6763526, upper bound: 0.6809986
IS_A1_B2_B1_B2_A1_A1, status: Status.VERIFIED, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6623665, upper bound: 0.6587198
IS_A1_B2_B1_B2_A1_A2, status: Status.VERIFIED, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6664660, upper bound: 0.6709246
IS_A1_B2_B1_B2_A2_B1, status: Status.VERIFIED, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6660722, upper bound: 0.6749682
IS_A1_B2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6663653, upper bound: 0.6721294
IS_A1_B2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6782956, upper bound: 0.6895300
IS_A1_B2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6782956, upper bound: 0.6895318
IS_A1_B2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6784391, upper bound: 0.6856586
IS_A1_B2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 18.10
Output dim: 6, lower bound: -0.6784391, upper bound: 0.6856586
IS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6782956, upper bound: 0.6854030
IS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6784391, upper bound: 0.6819416
IS_A2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6808680, upper bound: 0.6802604
IS_A2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6809988, upper bound: 0.6763548
IS_A2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6808680, upper bound: 0.6761300
IS_A2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6809988, upper bound: 0.6726409
IS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6885980, upper bound: 0.6709813
IS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6847036, upper bound: 0.6711258
IS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6844863, upper bound: 0.6709814
IS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6809988, upper bound: 0.6711247
IS_A2_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6895296, upper bound: 0.6782962
IS_A2_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6856565, upper bound: 0.6784394
IS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6854029, upper bound: 0.6782957
IS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6819408, upper bound: 0.6784395
IS_A2_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6895298, upper bound: 0.6760332
IS_A2_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6856565, upper bound: 0.6761617
IS_A2_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6854029, upper bound: 0.6760345
IS_A2_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.10
Output dim: 6, lower bound: -0.6819408, upper bound: 0.6761625
Binary search (step 2): status=Status.UNKNOWN, k_low=3, k_high=3, k_mid=3, eps_mid=0.0117188, abs_max=1.2817935943603516
rel_dist={6: [-0.7326686905779951, 0.7326686039477615]}

## Binary Search with IS_dual Result
status: Status.VERIFIED
Maximum delta epsilon: 0.0078125
execution time: 2404.47 seconds
