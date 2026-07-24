## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.6321674314


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.2319031, -6.8680754, -9.2319031, -6.8680754, -1.5920982, 1.5920992)
1: (-12.2275009, -9.8786087, -12.2275009, -9.8786087, -1.7782269, 1.7782259)
2: (-8.2851877, -6.0800848, -8.2851877, -6.0800848, -1.9221311, 1.9221311)
3: (-10.4678621, -8.2051430, -10.4678621, -8.2051430, -1.9307184, 1.9307184)
4: (-4.8220549, -2.7002640, -4.8220549, -2.7002640, -1.6834741, 1.6834741)
5: (-2.4755304, -0.3218992, -2.4755304, -0.3218992, -1.6719646, 1.6719651)
6: (9.4831448, 11.4725485, 9.4831448, 11.4725485, -1.1016169, 1.1016171)
7: (-21.0271072, -18.1547089, -21.0271072, -18.1547089, -1.7424874, 1.7424872)
8: (-2.3611345, -0.4984207, -2.3611345, -0.4984207, -1.3420033, 1.3420031)
9: (-13.2655563, -11.0138893, -13.2655563, -11.0138893, -1.5054874, 1.5054879)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.57 + 48.31 = 70.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.6334342, upper bound: 0.6334355

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5735
type: A, layer: 1, pos: 5846
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 6168
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 119

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 5735

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334303, upper bound: 0.6287730
time: 7.20 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334303, upper bound: 0.6334314
time: 5.37 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 12.79 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 12.79
Output dim: 6, lower bound: -0.6334303, upper bound: 0.6287730
NS_A2, status: Status.UNKNOWN, split count: 1, time: 12.79
Output dim: 6, lower bound: -0.6334303, upper bound: 0.6334314

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -9.2244301, -6.8706088, -9.2274294, -6.8695865, -1.5797558, 1.5820775
1: -12.2128296, -9.8791866, -12.2187195, -9.8789539, -1.7629528, 1.7686672
2: -8.2839775, -6.0824966, -8.2844677, -6.0815310, -1.9141960, 1.9125938
3: -10.4498911, -8.2052660, -10.4571009, -8.2052155, -1.9113288, 1.9190369
4: -4.8052850, -2.7016330, -4.8120165, -2.7010770, -1.6608458, 1.6678734
5: -2.4749694, -0.3344632, -2.4751964, -0.3294178, -1.6640310, 1.6592398
6: 9.4841928, 11.4543991, 9.4837666, 11.4616871, -1.0896842, 1.0827794
7: -21.0080624, -18.1568222, -21.0157127, -18.1559715, -1.7221155, 1.7290082
8: -2.3590121, -0.5157762, -2.3598700, -0.5088048, -1.3273063, 1.3219218
9: -13.2417831, -11.0142956, -13.2513247, -11.0141315, -1.4815292, 1.4908648

Time for backsubstitution: 21.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 1, pos: 5846

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6290153, upper bound: 0.6283444
time: 8.25 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334149, upper bound: 0.6287580
time: 7.05 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -9.2333059, -6.8553367, -9.2318945, -6.8680778, -1.5885706, 1.6034713
1: -12.2360306, -9.8608837, -12.2274981, -9.8786077, -1.7776709, 1.7961378
2: -8.2952499, -6.0780258, -8.2851849, -6.0800896, -1.9299073, 1.9233503
3: -10.4745073, -8.1914921, -10.4678583, -8.2051449, -1.9308720, 1.9443245
4: -4.8295035, -2.6798084, -4.8220377, -2.7002630, -1.6814322, 1.7020721
5: -2.4873419, -0.3195072, -2.4755280, -0.3219023, -1.6838598, 1.6677656
6: 9.4551535, 11.4735546, 9.4831438, 11.4725399, -1.1263287, 1.0905356
7: -21.0281334, -18.1333447, -21.0270958, -18.1547108, -1.7310863, 1.7613299
8: -2.3786216, -0.4956751, -2.3611336, -0.4984264, -1.3533473, 1.3322582
9: -13.2677460, -10.9873409, -13.2655478, -11.0138874, -1.4915085, 1.5258331

Time for backsubstitution: 21.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5846
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 119

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 5846

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6290153, upper bound: 0.6330028
time: 6.34 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334149, upper bound: 0.6334149
time: 6.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 34.80 seconds
NS_A1_B1, status: Status.VERIFIED, split count: 2, time: 34.80
Output dim: 6, lower bound: -0.6290153, upper bound: 0.6283444
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 34.80
Output dim: 6, lower bound: -0.6334149, upper bound: 0.6287580
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 34.80
Output dim: 6, lower bound: -0.6290153, upper bound: 0.6330028
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 34.80
Output dim: 6, lower bound: -0.6334149, upper bound: 0.6334149

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -9.2244320, -6.8706088, -9.2274294, -6.8695898, -1.5618782, 1.5820775
1: -12.2128305, -9.8791857, -12.2187176, -9.8789549, -1.7529111, 1.7683558
2: -8.2839737, -6.0824952, -8.2844629, -6.0815325, -1.9141936, 1.8915014
3: -10.4498901, -8.2052679, -10.4570980, -8.2052193, -1.8630004, 1.9175148
4: -4.8052845, -2.7016339, -4.8120089, -2.7010779, -1.6581435, 1.6348310
5: -2.4749684, -0.3344623, -2.4751961, -0.3294222, -1.6560535, 1.6592393
6: 9.4841909, 11.4543962, 9.4837666, 11.4616861, -1.0621190, 1.0827773
7: -21.0080643, -18.1568222, -21.0157089, -18.1559677, -1.7149200, 1.6544814
8: -2.3590107, -0.5157785, -2.3598695, -0.5088120, -1.2611976, 1.3121474
9: -13.2417812, -11.0142946, -13.2513218, -11.0141325, -1.4796624, 1.4663982

Time for backsubstitution: 22.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 6168
type: A, layer: 1, pos: 5846
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329957, upper bound: 0.6249591
time: 8.46 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334132, upper bound: 0.6287557
time: 16.86 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -9.2315216, -6.8711424, -9.2212133, -6.8972893, -1.5583239, 1.5781159
1: -12.2330723, -9.8665218, -12.2175808, -9.8890276, -1.7643147, 1.7801113
2: -8.2756176, -6.0797563, -8.2492933, -6.0933518, -1.8969893, 1.8848267
3: -10.4725723, -8.2173996, -10.4495258, -8.2518482, -1.8826313, 1.8974905
4: -4.8059745, -2.6807363, -4.7777276, -2.7111650, -1.6421819, 1.6580462
5: -2.4860582, -0.3319581, -2.4663732, -0.3446810, -1.6603460, 1.6478047
6: 9.4574642, 11.4590492, 9.4960775, 11.4464302, -1.0982232, 1.0635965
7: -20.9899654, -18.1352577, -20.9583454, -18.1806259, -1.6635876, 1.6913621
8: -2.3767514, -0.5330801, -2.3359089, -0.5660458, -1.2840867, 1.2636936
9: -13.2515764, -10.9881945, -13.2358875, -11.0226603, -1.4647112, 1.4959722

Time for backsubstitution: 22.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 6168
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 5846
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 119

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6285880, upper bound: 0.6291916
time: 5.22 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6290137, upper bound: 0.6329991
time: 5.76 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -9.2333050, -6.8553410, -9.2318945, -6.8680778, -1.5706940, 1.6034689
1: -12.2360315, -9.8608828, -12.2274971, -9.8786087, -1.7675915, 1.7958231
2: -8.2952499, -6.0780249, -8.2851830, -6.0800896, -1.9299054, 1.9022593
3: -10.4745064, -8.1914940, -10.4678593, -8.2051477, -1.8825355, 1.9400103
4: -4.8295016, -2.6798091, -4.8220320, -2.7002633, -1.6787271, 1.6641603
5: -2.4873414, -0.3195091, -2.4755297, -0.3219048, -1.6758823, 1.6677651
6: 9.4551563, 11.4735527, 9.4831429, 11.4725380, -1.0983946, 1.0905342
7: -21.0281334, -18.1333408, -21.0270901, -18.1547127, -1.7238674, 1.6858633
8: -2.3786211, -0.4956779, -2.3611345, -0.4984345, -1.2862916, 1.3229666
9: -13.2677460, -10.9873381, -13.2655468, -11.0138874, -1.4896383, 1.4995825

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6113
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 6168
type: A, layer: 1, pos: 5846
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 119

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 1, pos: 6113

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329957, upper bound: 0.6296155
time: 7.42 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334132, upper bound: 0.6334132
time: 7.66 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 37.38 seconds
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 37.38
Output dim: 6, lower bound: -0.6329957, upper bound: 0.6249591
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 37.38
Output dim: 6, lower bound: -0.6334132, upper bound: 0.6287557
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 37.38
Output dim: 6, lower bound: -0.6285880, upper bound: 0.6291916
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 37.38
Output dim: 6, lower bound: -0.6290137, upper bound: 0.6329991
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 37.38
Output dim: 6, lower bound: -0.6329957, upper bound: 0.6296155
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 37.38
Output dim: 6, lower bound: -0.6334132, upper bound: 0.6334132

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -9.2208309, -6.8815179, -9.2253065, -6.8760996, -1.5529814, 1.5698571
1: -12.2049427, -9.8813591, -12.2139997, -9.8802347, -1.7440157, 1.7619181
2: -8.2751274, -6.0854077, -8.2791777, -6.0832467, -1.8950496, 1.8753881
3: -10.4217033, -8.2082767, -10.4402332, -8.2069979, -1.8322406, 1.8980064
4: -4.7998195, -2.7267640, -4.8087835, -2.7160945, -1.6400652, 1.6080465
5: -2.4723818, -0.3595376, -2.4736800, -0.3443979, -1.6394348, 1.6333508
6: 9.4855471, 11.4412155, 9.4845705, 11.4538145, -1.0533054, 1.0690231
7: -21.0005207, -18.1592026, -21.0111904, -18.1573696, -1.7052970, 1.6469674
8: -2.3552985, -0.5207167, -2.3576374, -0.5117674, -1.2530622, 1.3036926
9: -13.2237835, -11.0165520, -13.2405500, -11.0154667, -1.4601340, 1.4530053

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 119

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6292893, upper bound: 0.6249545
time: 10.10 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329925, upper bound: 0.6249547
time: 6.32 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -9.2344923, -6.8694534, -9.2274284, -6.8695955, -1.5726762, 1.5831447
1: -12.2144775, -9.8723240, -12.2187109, -9.8789539, -1.7538786, 1.7756381
2: -8.2856941, -6.0732627, -8.2844563, -6.0815334, -1.9101734, 1.9030266
3: -10.4551268, -8.1860342, -10.4570866, -8.2052231, -1.8637791, 1.9281998
4: -4.8302822, -2.6998396, -4.8120084, -2.7010906, -1.6697569, 1.6351624
5: -2.4989192, -0.3329073, -2.4751961, -0.3294308, -1.6802416, 1.6596284
6: 9.4698744, 11.4549999, 9.4837685, 11.4616833, -1.0766799, 1.0756996
7: -21.0100632, -18.1520996, -21.0157032, -18.1559696, -1.7146950, 1.6592195
8: -2.3615165, -0.5149636, -2.3598666, -0.5088153, -1.2633123, 1.3125343
9: -13.2441616, -10.9961758, -13.2513151, -11.0141325, -1.4767938, 1.4820662

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 119

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6297047, upper bound: 0.6287541
time: 9.25 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334099, upper bound: 0.6287526
time: 5.16 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -9.2414799, -6.8699279, -9.2212105, -6.8973026, -1.5692787, 1.5792594
1: -12.2346573, -9.8596258, -12.2175732, -9.8890285, -1.7652049, 1.7874341
2: -8.2774200, -6.0705876, -8.2492886, -6.0933518, -1.8930426, 1.8964558
3: -10.4776173, -8.1981411, -10.4495134, -8.2518501, -1.8834009, 1.8996913
4: -4.8309546, -2.6789076, -4.7777257, -2.7111766, -1.6514583, 1.6584330
5: -2.5101287, -0.3304842, -2.4663696, -0.3446900, -1.6790018, 1.6481214
6: 9.4431391, 11.4596291, 9.4960775, 11.4464283, -1.0994043, 1.0564957
7: -20.9919109, -18.1304607, -20.9583435, -18.1806278, -1.6633053, 1.6930397
8: -2.3793697, -0.5322514, -2.3359051, -0.5660510, -1.2856169, 1.2641053
9: -13.2537680, -10.9700718, -13.2358818, -11.0226593, -1.4616809, 1.4982536

Time for backsubstitution: 21.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 119

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6253051, upper bound: 0.6329963
time: 7.05 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6290104, upper bound: 0.6329968
time: 4.70 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -9.2296009, -6.8662086, -9.2297707, -6.8745890, -1.5618296, 1.5913191
1: -12.2281570, -9.8630228, -12.2227774, -9.8798838, -1.7587175, 1.7894087
2: -8.2864342, -6.0810094, -8.2798967, -6.0818052, -1.9108233, 1.8861694
3: -10.4462891, -8.1944933, -10.4509621, -8.2069244, -1.8517990, 1.9172459
4: -4.8240042, -2.7048752, -4.8188033, -2.7152860, -1.6594934, 1.6373630
5: -2.4847806, -0.3445888, -2.4740224, -0.3368895, -1.6592684, 1.6418762
6: 9.4565125, 11.4603739, 9.4839478, 11.4646683, -1.0876927, 1.0767817
7: -21.0205765, -18.1356964, -21.0225677, -18.1560993, -1.7142487, 1.6775274
8: -2.3749747, -0.5005984, -2.3589039, -0.5013876, -1.2775662, 1.3145227
9: -13.2497501, -10.9895887, -13.2547626, -11.0152197, -1.4701109, 1.4837382

Time for backsubstitution: 21.86 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 119

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6292893, upper bound: 0.6296135
time: 7.82 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329925, upper bound: 0.6296122
time: 7.97 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -9.2432547, -6.8541203, -9.2318954, -6.8680859, -1.5816436, 1.6046143
1: -12.2376270, -9.8540020, -12.2274885, -9.8786125, -1.7684846, 1.8031435
2: -8.2970581, -6.0688648, -8.2851801, -6.0800920, -1.9259620, 1.9138861
3: -10.4795732, -8.1722651, -10.4678450, -8.2051487, -1.8833017, 1.9422321
4: -4.8544369, -2.6779780, -4.8220310, -2.7002764, -1.6868000, 1.6644425
5: -2.5114040, -0.3180322, -2.4755282, -0.3219144, -1.6929379, 1.6680846
6: 9.4408360, 11.4741306, 9.4831467, 11.4725361, -1.0995905, 1.0834343
7: -21.0300789, -18.1285515, -21.0270863, -18.1547146, -1.7235851, 1.6875415
8: -2.3812089, -0.4948492, -2.3611307, -0.4984365, -1.2878377, 1.3233788
9: -13.2699394, -10.9692240, -13.2655411, -11.0138884, -1.4866161, 1.5018749

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 6184
type: B, layer: 1, pos: 5735
type: B, layer: 1, pos: 6168
type: B, layer: 1, pos: 524
type: B, layer: 1, pos: 6113
type: B, layer: 1, pos: 932
type: B, layer: 1, pos: 6111
type: B, layer: 1, pos: 821
type: B, layer: 1, pos: 6213
type: B, layer: 1, pos: 5801
type: B, layer: 1, pos: 119

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 6184

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6297047, upper bound: 0.6334098
time: 6.75 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334099, upper bound: 0.6334097
time: 7.90 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 36.89 seconds
NS_A1_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 36.89
Output dim: 6, lower bound: -0.6292893, upper bound: 0.6249545
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.89
Output dim: 6, lower bound: -0.6329925, upper bound: 0.6249547
NS_A1_B2_A2_B1, status: Status.VERIFIED, split count: 4, time: 36.89
Output dim: 6, lower bound: -0.6297047, upper bound: 0.6287541
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.89
Output dim: 6, lower bound: -0.6334099, upper bound: 0.6287526
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.89
Output dim: 6, lower bound: -0.6253051, upper bound: 0.6329963
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.89
Output dim: 6, lower bound: -0.6290104, upper bound: 0.6329968
NS_A2_B2_A1_B1, status: Status.VERIFIED, split count: 4, time: 36.89
Output dim: 6, lower bound: -0.6292893, upper bound: 0.6296135
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 36.89
Output dim: 6, lower bound: -0.6329925, upper bound: 0.6296122
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 36.89
Output dim: 6, lower bound: -0.6297047, upper bound: 0.6334098
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 36.89
Output dim: 6, lower bound: -0.6334099, upper bound: 0.6334097

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -9.2208290, -6.8815203, -9.2253036, -6.8761015, -1.5526738, 1.5592165
1: -12.2049408, -9.8813591, -12.2139950, -9.8802357, -1.7440143, 1.7229018
2: -8.2751255, -6.0854073, -8.2791729, -6.0832486, -1.8900399, 1.8282156
3: -10.4217033, -8.2082787, -10.4402304, -8.2070036, -1.8013115, 1.8957200
4: -4.7998185, -2.7267683, -4.8087821, -2.7161016, -1.6178837, 1.6079135
5: -2.4723835, -0.3595392, -2.4736793, -0.3444026, -1.6242943, 1.6325970
6: 9.4855480, 11.4412136, 9.4845734, 11.4538155, -1.0336366, 1.0690219
7: -21.0005207, -18.1592026, -21.0111923, -18.1573715, -1.7024417, 1.6374331
8: -2.3552971, -0.5207157, -2.3576326, -0.5117688, -1.2530589, 1.2926333
9: -13.2237806, -11.0165520, -13.2405453, -11.0154667, -1.4601312, 1.4179125

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6168
type: A, layer: 1, pos: 5846
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 119

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6168

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329803, upper bound: 0.6223164
time: 7.88 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6329910, upper bound: 0.6249540
time: 5.26 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -9.2344913, -6.8694520, -9.2274246, -6.8695989, -1.5723600, 1.5725465
1: -12.2144756, -9.8723259, -12.2187014, -9.8789558, -1.7538738, 1.7366233
2: -8.2856941, -6.0732603, -8.2844486, -6.0815325, -1.9060616, 1.8558269
3: -10.4551268, -8.1860342, -10.4570847, -8.2052269, -1.8328590, 1.9206789
4: -4.8302817, -2.6998413, -4.8120079, -2.7010984, -1.6474662, 1.6350336
5: -2.4989197, -0.3329092, -2.4751954, -0.3294348, -1.6650333, 1.6588783
6: 9.4698753, 11.4549971, 9.4837704, 11.4616795, -1.0570130, 1.0756977
7: -21.0100651, -18.1520996, -21.0157051, -18.1559715, -1.7118473, 1.6496861
8: -2.3615150, -0.5149641, -2.3598609, -0.5088181, -1.2633090, 1.3014758
9: -13.2441607, -10.9961777, -13.2513113, -11.0141354, -1.4767923, 1.4463434

Time for backsubstitution: 21.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6168
type: A, layer: 1, pos: 5846
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 119

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6168

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6333981, upper bound: 0.6261210
time: 24.31 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6334084, upper bound: 0.6287505
time: 11.31 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -9.2345190, -6.8746624, -9.2071028, -6.9101138, -1.5483022, 1.5602555
1: -12.2121696, -9.8605766, -12.1729507, -9.9008713, -1.7262001, 1.7385745
2: -8.2489700, -6.0718989, -8.1936684, -6.1087770, -1.8393178, 1.8397021
3: -10.4719439, -8.2141266, -10.4292049, -8.2831936, -1.8457699, 1.8487287
4: -4.8299198, -2.6912901, -4.7699876, -2.7355983, -1.6250710, 1.6301355
5: -2.5086823, -0.3412123, -2.4583149, -0.3660551, -1.6552353, 1.6225157
6: 9.4462547, 11.4497910, 9.5094671, 11.4271679, -1.0770702, 1.0306802
7: -20.9859352, -18.1336136, -20.9463158, -18.1908550, -1.6435184, 1.6775060
8: -2.3690238, -0.5344982, -2.3147335, -0.5735188, -1.2528162, 1.2296028
9: -13.2334785, -10.9709492, -13.1959496, -11.0341091, -1.4273529, 1.4564414

Time for backsubstitution: 21.94 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 6168
type: A, layer: 1, pos: 524
type: A, layer: 1, pos: 5846
type: A, layer: 1, pos: 6184
type: A, layer: 1, pos: 6111
type: A, layer: 1, pos: 932
type: A, layer: 1, pos: 821
type: A, layer: 1, pos: 6213
type: A, layer: 1, pos: 5801
type: A, layer: 1, pos: 119

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 1, pos: 6168

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.6252933, upper bound: 0.6303624
time: 7.24 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.6253036, upper bound: 0.6329942
time: 5.41 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -9.2414780, -6.8699284, -9.2212067, -6.8973026, -1.5689812, 1.5686388
1: -12.2346544, -9.8596277, -12.2175665, -9.8890276, -1.7652025, 1.7484202
2: -8.2774181, -6.0705881, -8.2492828, -6.0933542, -1.8862004, 1.8494081
3: -10.4776192, -8.1981440, -10.4495115, -8.2518549, -1.8524890, 1.8921733
4: -4.8309555, -2.6789093, -4.7777252, -2.7111835, -1.6291585, 1.6535034
5: -2.5101280, -0.3304861, -2.4663708, -0.3446937, -1.6625214, 1.6473665
6: 9.4431391, 11.4596272, 9.4960794, 11.4464273, -1.0787621, 1.0564940
7: -20.9919128, -18.1304626, -20.9583416, -18.1806278, -1.6604514, 1.6832240
8: -2.3793683, -0.5322523, -2.3359003, -0.5660515, -1.2829580, 1.2530496
9: -13.2537642, -10.9700718, -13.2358751, -11.0226622, -1.4616776, 1.4625309

Time for backsubstitution: 22.07 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 70.89 + 537.54 = 608.42 seconds
