## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 817.226686863868


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-188.5958862, 756.7218628, -188.5958862, 756.7218628, -945.3177490, 945.3177490)
1: (-233.3839569, 855.3833008, -233.3839569, 855.3833008, -1088.7672119, 1088.7672119)
2: (-244.9800110, 867.1258545, -244.9800110, 867.1258545, -1112.1055908, 1112.1055908)
3: (-388.4680786, 915.8612061, -388.4680786, 915.8612061, -1304.3293457, 1304.3293457)
4: (-395.0017090, 881.3112793, -395.0017090, 881.3112793, -1276.3126221, 1276.3126221)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.50 + 1.88 = 3.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -817.2512044, upper bound: 817.2512044

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2497158, upper bound: 817.2489569
time: 0.59 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2490215, upper bound: 817.2490215
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.40 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 0, lower bound: -817.2497158, upper bound: 817.2489569
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.40
Output dim: 0, lower bound: -817.2490215, upper bound: 817.2490215

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -178.2229614, 715.3228149, -185.9003601, 745.9097900, -924.1325684, 901.2231445
1: -220.4872284, 808.5225830, -230.0360565, 843.1456299, -1063.6328125, 1038.5584717
2: -231.4507141, 819.4750366, -241.4617004, 854.6936646, -1086.1444092, 1060.9367676
3: -366.9500427, 865.4583740, -382.8678589, 902.7092285, -1269.6593018, 1248.3259277
4: -373.0900269, 832.7399902, -389.3070068, 868.6387329, -1241.7287598, 1222.0469971

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2490533, upper bound: 817.2461196
time: 0.59 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2490533, upper bound: 817.2470209
time: 0.66 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -242.0322266, 981.8877563, -178.9747162, 716.8466797, -958.8789062, 1158.6240234
1: -299.3828735, 1109.1506348, -221.4718323, 810.2425537, -1109.6251221, 1327.8457031
2: -314.1368408, 1124.1547852, -232.2762299, 821.4516602, -1135.5885010, 1353.9036865
3: -501.5453186, 1188.1212158, -368.4362183, 867.7747803, -1369.3197021, 1554.1080322
4: -510.6184692, 1141.5734863, -374.5950012, 834.9650879, -1345.5834961, 1514.5268555

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2468930, upper bound: 817.2459062
time: 1.06 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2469694, upper bound: 817.2469694
time: 0.72 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.25 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -817.2490533, upper bound: 817.2461196
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -817.2490533, upper bound: 817.2470209
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -817.2468930, upper bound: 817.2459062
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.25
Output dim: 0, lower bound: -817.2469694, upper bound: 817.2469694

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -168.0992889, 672.8702393, -170.5336761, 681.4674072, -849.5667114, 843.4039307
1: -207.9964447, 760.4459229, -211.0703278, 770.1578979, -978.1542358, 971.5161743
2: -218.1373596, 770.9128418, -221.2469940, 780.9802856, -999.1176758, 992.1598511
3: -345.8485107, 814.2531738, -350.8171997, 824.9835205, -1170.8320312, 1165.0703125
4: -351.5947266, 783.3320923, -356.6699829, 793.6375732, -1145.2322998, 1140.0020752

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2170587, upper bound: 817.1751433
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1927036, upper bound: 817.1721979
time: 0.62 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -175.6127472, 704.9499512, -190.5993347, 763.5314941, -939.1442261, 895.5492554
1: -217.2399597, 796.7643433, -235.8658447, 863.0693970, -1080.3093262, 1032.6301270
2: -228.0120087, 807.5507812, -247.3920593, 875.0758667, -1103.0878906, 1054.9428711
3: -361.5274353, 852.8099976, -392.3591614, 924.2033691, -1285.7308350, 1245.1691895
4: -367.5856018, 820.5524292, -398.7913208, 889.5356445, -1257.1210938, 1219.3437500

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1579605, upper bound: 817.0756471
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1528542, upper bound: 817.0756259
time: 0.72 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -232.3769989, 940.8505859, -165.0385437, 658.5051880, -890.8822021, 1103.6429443
1: -287.4270935, 1062.6368408, -204.2720032, 744.1818848, -1031.6090088, 1264.1331787
2: -301.3566895, 1077.1998291, -213.9452362, 754.7273560, -1056.0837402, 1288.5771484
3: -481.2124634, 1138.5566406, -339.4273682, 797.4602051, -1278.6726074, 1475.5422363
4: -490.0005188, 1093.7052002, -345.0443420, 767.0853271, -1257.0858154, 1437.1625977

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2040077, upper bound: 817.1570927
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.0733670, upper bound: 817.1186823
time: 0.71 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -238.2202911, 966.6730957, -182.0428619, 727.9918213, -966.2120972, 1146.4766846
1: -294.6368408, 1091.8754883, -225.2951202, 822.7810669, -1117.4179688, 1314.4113770
2: -309.1185608, 1106.6712646, -236.0693054, 834.4246826, -1143.5432129, 1340.2388916
3: -493.6178589, 1169.5651855, -374.5736694, 881.3236694, -1374.9415283, 1541.6959229
4: -502.5639648, 1123.7424316, -380.6664124, 848.2000732, -1350.7640381, 1502.7862549

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.1506354, upper bound: 817.0750941
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.0712626, upper bound: 817.0712626
time: 0.63 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.80 seconds
NS_A1_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -817.2170587, upper bound: 817.1751433
NS_A1_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -817.1927036, upper bound: 817.1721979
NS_A1_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -817.1579605, upper bound: 817.0756471
NS_A1_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -817.1528542, upper bound: 817.0756259
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -817.2040077, upper bound: 817.1570927
NS_A2_B1_A2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -817.0733670, upper bound: 817.1186823
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -817.1506354, upper bound: 817.0750941
NS_A2_B2_A2, status: Status.VERIFIED, split count: 3, time: 2.80
Output dim: 0, lower bound: -817.0712626, upper bound: 817.0712626

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.38 + 18.67 = 22.05 seconds
