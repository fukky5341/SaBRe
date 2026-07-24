## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 2027.3678997182642


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-796.8801270, 1389.0648193, -796.8801270, 1389.0648193, -2185.9448242, 2185.9448242)
1: (-722.3177490, 1251.5062256, -722.3177490, 1251.5062256, -1973.8239746, 1973.8239746)
2: (-632.7144775, 1318.1571045, -632.7144775, 1318.1571045, -1950.8714600, 1950.8714600)
3: (-972.6072998, 1298.1092529, -972.6072998, 1298.1092529, -2270.7165527, 2270.7165527)
4: (-767.1251831, 1415.1643066, -767.1251831, 1415.1643066, -2182.2895508, 2182.2895508)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.55 + 2.14 = 3.69 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -2027.3881736, upper bound: 2027.3881736

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 38

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3744085, upper bound: 2027.3778941
time: 0.99 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3743616, upper bound: 2027.3743616
time: 0.72 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.83 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 0, lower bound: -2027.3744085, upper bound: 2027.3778941
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.83
Output dim: 0, lower bound: -2027.3743616, upper bound: 2027.3743616

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -721.5062256, 1255.9519043, -790.2480469, 1377.3643799, -2098.8701172, 2046.1999512
1: -654.1708374, 1131.3173828, -716.3209229, 1240.9368896, -1895.1076660, 1847.6383057
2: -573.0798950, 1192.2563477, -627.4632568, 1307.0905762, -1880.1699219, 1819.7196045
3: -881.2708130, 1173.7810059, -964.5790405, 1287.1730957, -2168.4431152, 2138.3601074
4: -695.0112915, 1280.7889404, -760.7811279, 1403.3537598, -2098.3647461, 2041.5697021

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3743347, upper bound: 2027.3771520
time: 0.76 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3742113, upper bound: 2027.3775539
time: 0.77 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1100.4594727, 1910.4429932, -768.8742676, 1340.8829346, -2441.3422852, 2679.3173828
1: -997.4267578, 1718.9854736, -697.0084839, 1208.0938721, -2205.5205078, 2415.9938965
2: -874.0090332, 1811.3371582, -610.5125122, 1272.3453369, -2146.3544922, 2421.8496094
3: -1343.6478271, 1785.8741455, -938.4584351, 1252.8962402, -2596.5439453, 2724.3322754
4: -1061.2701416, 1947.5059814, -740.1497192, 1365.7956543, -2427.0659180, 2687.6555176

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 12

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3742970, upper bound: 2027.3741239
time: 0.80 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3741905, upper bound: 2027.3741905
time: 0.86 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.21 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -2027.3743347, upper bound: 2027.3771520
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -2027.3742113, upper bound: 2027.3775539
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -2027.3742970, upper bound: 2027.3741239
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.21
Output dim: 0, lower bound: -2027.3741905, upper bound: 2027.3741905

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -712.3112183, 1239.9669189, -765.6063843, 1334.4876709, -2046.7988281, 2005.5732422
1: -645.8156128, 1116.9400635, -693.9492798, 1202.3587646, -1848.1743164, 1810.8892822
2: -565.7797852, 1177.1224365, -607.9236450, 1266.4704590, -1832.2501221, 1785.0460205
3: -870.0721436, 1158.8315430, -934.5647583, 1247.0798340, -2117.1518555, 2093.3962402
4: -686.1289673, 1264.5434570, -737.0458984, 1359.7497559, -2045.8786621, 2001.5889893

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3684144, upper bound: 2027.3687624
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3738442, upper bound: 2027.3750617
time: 0.80 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -700.1211548, 1219.3044434, -813.1269531, 1417.4898682, -2117.6110840, 2032.4313965
1: -634.7749634, 1098.4023438, -736.6763306, 1277.8822021, -1912.6572266, 1835.0786133
2: -556.0520630, 1157.3731689, -645.3338623, 1345.3989258, -1901.4509277, 1802.7070312
3: -855.0480347, 1139.4180908, -992.0730591, 1325.0286865, -2180.0766602, 2131.4907227
4: -674.2498779, 1243.1124268, -782.1148682, 1444.3238525, -2118.5734863, 2025.2271729

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735478, upper bound: 2027.3775539
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735558, upper bound: 2027.3775530
time: 0.75 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1092.2196045, 1896.0554199, -743.2986450, 1296.3969727, -2388.6164551, 2639.3540039
1: -989.9337158, 1706.0278320, -673.7861938, 1168.0495605, -2157.9833984, 2379.8139648
2: -867.4680786, 1797.7030029, -590.2274780, 1230.1796875, -2097.6477051, 2387.9301758
3: -1333.6103516, 1772.4584961, -907.2880859, 1211.2625732, -2544.8728027, 2679.7465820
4: -1053.3544922, 1932.8840332, -715.5115967, 1320.5117188, -2373.8662109, 2648.3955078

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3741239, upper bound: 2027.3741239
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3741239, upper bound: 2027.3741239
time: 0.85 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1074.3022461, 1865.8878174, -787.2210693, 1373.3006592, -2447.6030273, 2653.1079102
1: -973.7376099, 1679.0377197, -713.2199097, 1238.0974121, -2211.8349609, 2392.2575684
2: -853.1992798, 1768.9493408, -624.7574463, 1303.1580811, -2156.3564453, 2393.7067871
3: -1311.4615479, 1744.1551514, -960.1053467, 1283.6247559, -2595.0864258, 2704.2602539
4: -1035.9270020, 1901.5523682, -757.1790771, 1398.5070801, -2434.4340820, 2658.7314453

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735243, upper bound: 2027.3741905
time: 0.94 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735309, upper bound: 2027.3735309
time: 0.79 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.30 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -2027.3684144, upper bound: 2027.3687624
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -2027.3738442, upper bound: 2027.3750617
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -2027.3735478, upper bound: 2027.3775539
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -2027.3735558, upper bound: 2027.3775530
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -2027.3741239, upper bound: 2027.3741239
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -2027.3741239, upper bound: 2027.3741239
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -2027.3735243, upper bound: 2027.3741905
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.30
Output dim: 0, lower bound: -2027.3735309, upper bound: 2027.3735309

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -637.8801270, 1109.7185059, -722.1438599, 1258.5983887, -1896.4783936, 1831.8623047
1: -578.3330078, 998.8895874, -654.5883179, 1133.7313232, -1712.0643311, 1653.4774170
2: -506.8007812, 1053.1241455, -573.3798828, 1194.3325195, -1701.1333008, 1626.5040283
3: -779.2620850, 1036.5429688, -881.7260742, 1175.9543457, -1955.2163086, 1918.2689209
4: -614.8667603, 1131.7675781, -695.3213501, 1282.3955078, -1897.2622070, 1827.0888672

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3684144, upper bound: 2027.3687624
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3684144, upper bound: 2027.3687624
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -674.9192505, 1173.9262695, -738.1438599, 1286.3240967, -1961.2434082, 1912.0700684
1: -611.7370605, 1057.7625732, -668.9531860, 1159.0845947, -1770.8216553, 1726.7158203
2: -535.7213745, 1114.9628906, -585.9592896, 1220.9626465, -1756.6840820, 1700.9221191
3: -824.5175781, 1097.6024170, -901.0018311, 1202.2202148, -2026.7377930, 1998.6042480
4: -649.6031494, 1197.8837891, -710.4519653, 1310.8748779, -1960.4780273, 1908.3356934

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3738442, upper bound: 2027.3750617
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3738442, upper bound: 2027.3750617
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -653.2814941, 1137.4301758, -783.9171753, 1366.3375244, -2019.6190186, 1921.3474121
1: -592.1227417, 1024.5612793, -710.1151123, 1231.7940674, -1823.9167480, 1734.6762695
2: -518.7032471, 1079.5148926, -622.0942993, 1296.7884521, -1815.4916992, 1701.6091309
3: -797.7719116, 1062.8048096, -956.3430786, 1277.1940918, -2074.9660645, 2019.1479492
4: -628.9107666, 1159.5383301, -753.8865356, 1392.1647949, -2021.0755615, 1913.4248047

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735478, upper bound: 2027.3775539
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735478, upper bound: 2027.3775539
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -783.1141357, 1365.0209961, -796.2908325, 1388.1423340, -2171.2563477, 2161.3117676
1: -709.4626465, 1230.6818848, -721.3884888, 1251.3739014, -1960.8365479, 1952.0703125
2: -621.4654541, 1296.1308594, -631.9828491, 1317.5030518, -1938.9683838, 1928.1136475
3: -956.1210938, 1275.8999023, -971.6052246, 1297.5543213, -2253.6750488, 2247.5051270
4: -753.1934204, 1391.9265137, -765.9779663, 1414.4110107, -2167.6044922, 2157.9045410

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735558, upper bound: 2027.3775530
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735558, upper bound: 2027.3775530
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1078.3455811, 1871.8929443, -743.2986450, 1296.3969727, -2374.7426758, 2615.1916504
1: -977.3076782, 1684.2636719, -673.7861938, 1168.0495605, -2145.3571777, 2358.0498047
2: -856.4492798, 1774.7985840, -590.2274780, 1230.1796875, -2086.6289062, 2365.0256348
3: -1316.6868896, 1749.9127197, -907.2880859, 1211.2625732, -2527.9494629, 2657.2006836
4: -1040.0201416, 1908.3026123, -715.5115967, 1320.5117188, -2360.5317383, 2623.8142090

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3718814, upper bound: 2027.3722323
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3742623, upper bound: 2027.3740888
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1104.8387451, 1919.7547607, -743.2986450, 1296.3969727, -2401.2353516, 2663.0534668
1: -1001.1018677, 1728.3038330, -673.7861938, 1168.0495605, -2169.1513672, 2402.0900879
2: -877.1681519, 1820.2088623, -590.2274780, 1230.1796875, -2107.3479004, 2410.4362793
3: -1348.0771484, 1795.2979736, -907.2880859, 1211.2625732, -2559.3395996, 2702.5859375
4: -1064.9229736, 1956.0797119, -715.5115967, 1320.5117188, -2385.4345703, 2671.5913086

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3718814, upper bound: 2027.3722323
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3742623, upper bound: 2027.3740888
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1023.6257324, 1777.5255127, -757.3590088, 1321.0552979, -2344.6809082, 2534.8845215
1: -927.7369995, 1599.2441406, -686.0563965, 1190.9783936, -2118.7153320, 2285.2998047
2: -812.9138184, 1684.8997803, -600.9856567, 1253.4468994, -2066.3608398, 2285.8845215
3: -1249.7152100, 1661.3923340, -923.5946045, 1234.7169189, -2484.4318848, 2584.9865723
4: -987.0911865, 1811.3540039, -728.3549194, 1345.1650391, -2332.2561035, 2539.7084961

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 6

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3698041, upper bound: 2027.3710682
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3734970, upper bound: 2027.3741651
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1106.1820068, 1921.3311768, -765.2087402, 1334.7788086, -2440.9604492, 2686.5395508
1: -1002.0878296, 1729.5379639, -693.2500610, 1203.2049561, -2205.2927246, 2422.7880859
2: -877.7501831, 1821.8784180, -607.3074341, 1266.4328613, -2144.1826172, 2429.1857910
3: -1349.2705078, 1796.6441650, -933.3818359, 1247.5037842, -2596.7744141, 2730.0258789
4: -1065.8801270, 1957.6231689, -736.0888062, 1359.1600342, -2425.0400391, 2693.7119141

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3698667, upper bound: 2027.3706720
time: 0.85 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735106, upper bound: 2027.3735106
time: 0.63 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.39 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3684144, upper bound: 2027.3687624
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3684144, upper bound: 2027.3687624
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3738442, upper bound: 2027.3750617
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3738442, upper bound: 2027.3750617
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3735478, upper bound: 2027.3775539
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3735478, upper bound: 2027.3775539
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3735558, upper bound: 2027.3775530
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3735558, upper bound: 2027.3775530
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3718814, upper bound: 2027.3722323
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3742623, upper bound: 2027.3740888
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3718814, upper bound: 2027.3722323
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3742623, upper bound: 2027.3740888
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3698041, upper bound: 2027.3710682
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3734970, upper bound: 2027.3741651
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3698667, upper bound: 2027.3706720
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.39
Output dim: 0, lower bound: -2027.3735106, upper bound: 2027.3735106

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -637.8801270, 1109.7185059, -657.1621704, 1143.5322266, -1781.4122314, 1766.8806152
1: -578.3330078, 998.8895874, -595.8187866, 1029.7995605, -1608.1325684, 1594.7078857
2: -506.8007812, 1053.1241455, -522.1062622, 1085.5240479, -1592.3248291, 1575.2304688
3: -779.2620850, 1036.5429688, -802.8892212, 1068.5133057, -1847.7752686, 1839.4321289
4: -614.8667603, 1131.7675781, -633.2908936, 1166.2967529, -1781.1634521, 1765.0582275

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3673258, upper bound: 2027.3656738
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3586977, upper bound: 2027.3669522
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3668108, upper bound: 2027.3679596
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3675500, upper bound: 2027.3672824
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -637.8801270, 1109.7185059, -1036.1872559, 1797.7734375, -2435.6535645, 2145.9057617
1: -578.3330078, 998.8895874, -939.1342773, 1617.2210693, -2195.5541992, 1938.0236816
2: -506.8007812, 1053.1241455, -823.0169678, 1704.4317627, -2211.2324219, 1876.1409912
3: -779.2620850, 1036.5429688, -1265.5798340, 1680.7491455, -2460.0112305, 2302.1228027
4: -614.8667603, 1131.7675781, -999.6131592, 1833.1767578, -2448.0429688, 2131.3808594

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3673258, upper bound: 2027.3656738
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 46

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 1

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3586977, upper bound: 2027.3669522
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3668108, upper bound: 2027.3679596
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3675500, upper bound: 2027.3672824
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -674.9192505, 1173.9262695, -669.5109863, 1165.2637939, -1840.1831055, 1843.4372559
1: -611.7370605, 1057.7625732, -606.8727417, 1049.7790527, -1661.5161133, 1664.6350098
2: -535.7213745, 1114.9628906, -531.6268311, 1106.4914551, -1642.2128906, 1646.5897217
3: -824.5175781, 1097.6024170, -817.8705444, 1089.1119385, -1913.6295166, 1915.4729004
4: -649.6031494, 1197.8837891, -644.6245117, 1188.6802979, -1838.2834473, 1842.5083008

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3717880, upper bound: 2027.3729319
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3738252, upper bound: 2027.3750405
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -674.9192505, 1173.9262695, -1038.8455811, 1803.3074951, -2478.2265625, 2212.7719727
1: -611.7370605, 1057.7625732, -941.3206787, 1622.5941162, -2234.3310547, 1999.0831299
2: -535.7213745, 1114.9628906, -824.8291626, 1709.7915039, -2245.5124512, 1939.7919922
3: -824.5175781, 1097.6024170, -1268.3999023, 1685.8870850, -2510.4045410, 2366.0017090
4: -649.6031494, 1197.8837891, -1001.7371826, 1838.3096924, -2487.9118652, 2199.6210938

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3717880, upper bound: 2027.3729319
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3738252, upper bound: 2027.3750405
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -653.2814941, 1137.4301758, -715.9700928, 1246.4633789, -1899.7448730, 1853.4001465
1: -592.1227417, 1024.5612793, -648.6909180, 1123.4781494, -1715.6008301, 1673.2520752
2: -518.7032471, 1079.5148926, -568.3252563, 1183.4324951, -1702.1357422, 1647.8400879
3: -797.7719116, 1062.8048096, -874.1071167, 1165.2062988, -1962.9780273, 1936.9117432
4: -628.9107666, 1159.5383301, -688.8331909, 1271.1635742, -1900.0742188, 1848.3714600

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3712660, upper bound: 2027.3740469
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735050, upper bound: 2027.3775352
time: 0.75 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -653.2814941, 1137.4301758, -1074.3867188, 1866.6458740, -2519.9272461, 2211.8166504
1: -592.1227417, 1024.5612793, -973.4102783, 1680.4152832, -2272.5380859, 1997.9715576
2: -518.7032471, 1079.5148926, -852.8859863, 1769.6728516, -2288.3759766, 1932.4008789
3: -797.7719116, 1062.8048096, -1310.8395996, 1745.6102295, -2543.3820801, 2373.6437988
4: -628.9107666, 1159.5383301, -1035.4647217, 1901.7839355, -2530.6948242, 2195.0026855

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3712660, upper bound: 2027.3740473
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735050, upper bound: 2027.3775352
time: 0.92 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -783.1141357, 1365.0209961, -729.8052979, 1270.7463379, -2053.8603516, 2094.8261719
1: -709.4626465, 1230.6818848, -661.3066406, 1145.3089600, -1854.7716064, 1891.9884033
2: -621.4654541, 1296.1308594, -579.3999634, 1206.4986572, -1827.9639893, 1875.5307617
3: -956.1210938, 1275.8999023, -891.0460815, 1187.9161377, -2144.0366211, 2166.9460449
4: -753.1934204, 1391.9265137, -702.3972168, 1295.8898926, -2049.0832520, 2094.3234863

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3713500, upper bound: 2027.3750743
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3775352
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -783.1141357, 1365.0209961, -1082.1250000, 1880.0640869, -2663.1779785, 2447.1459961
1: -709.4626465, 1230.6818848, -980.5483398, 1692.2336426, -2401.6960449, 2211.2302246
2: -621.4654541, 1296.1308594, -859.2097168, 1782.3580322, -2403.8234863, 2155.3405762
3: -956.1210938, 1275.8999023, -1320.6121826, 1757.9630127, -2714.0834961, 2596.5122070
4: -753.1934204, 1391.9265137, -1043.2391357, 1915.5864258, -2668.7797852, 2435.1655273

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3713500, upper bound: 2027.3750743
time: 1.01 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3775352
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1069.0838623, 1855.6467285, -721.3527222, 1258.0163574, -2327.1000977, 2576.9995117
1: -968.9022827, 1669.6008301, -653.8937988, 1133.4575195, -2102.3598633, 2323.4946289
2: -849.1134033, 1759.4227295, -572.8470459, 1193.9019775, -2043.0153809, 2332.2697754
3: -1305.4583740, 1734.7437744, -880.7182617, 1175.3685303, -2480.8269043, 2615.4619141
4: -1031.1402588, 1891.8607178, -694.4357910, 1281.6772461, -2312.8173828, 2586.2963867

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3721949, upper bound: 2027.3734037
time: 1.01 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3721800, upper bound: 2027.3729540
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1063.3536377, 1846.0045166, -737.3591919, 1286.0162354, -2349.3698730, 2583.3635254
1: -963.7196045, 1660.8048096, -668.3278198, 1158.5812988, -2122.3007812, 2329.1325684
2: -844.5875854, 1750.1046143, -585.5212402, 1220.2298584, -2064.8173828, 2335.6259766
3: -1298.3826904, 1725.5546875, -899.8977661, 1201.4741211, -2499.8566895, 2625.4523926
4: -1025.6865234, 1881.7239990, -709.7874756, 1309.8385010, -2335.5249023, 2591.5114746

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3733707, upper bound: 2027.3741369
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3733119, upper bound: 2027.3733119
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1095.2275391, 1902.9327393, -721.3527222, 1258.0163574, -2353.2436523, 2624.2851562
1: -992.3740845, 1713.1011963, -653.8937988, 1133.4575195, -2125.8315430, 2366.9951172
2: -869.5451050, 1804.2592773, -572.8470459, 1193.9019775, -2063.4470215, 2377.1059570
3: -1336.4135742, 1779.5627441, -880.7182617, 1175.3685303, -2511.7822266, 2660.2810059
4: -1055.6922607, 1939.0183105, -694.4357910, 1281.6772461, -2337.3693848, 2633.4541016

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3090327, upper bound: 2027.3411010
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3080446, upper bound: 2027.3363445
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1090.5444336, 1894.9520264, -737.3591919, 1286.0162354, -2376.5605469, 2632.3107910
1: -988.1520386, 1705.8695068, -668.3278198, 1158.5812988, -2146.7333984, 2374.1972656
2: -865.8778076, 1796.6273193, -585.5212402, 1220.2298584, -2086.1076660, 2382.1484375
3: -1330.6370850, 1772.0264893, -899.8977661, 1201.4741211, -2532.1113281, 2671.9243164
4: -1051.2790527, 1930.7238770, -709.7874756, 1309.8385010, -2361.1176758, 2640.5112305

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3724056, upper bound: 2027.3719612
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3724331, upper bound: 2027.3722925
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1013.3455200, 1759.5007324, -734.9179077, 1281.8675537, -2295.2128906, 2494.4182129
1: -918.4163208, 1582.9605713, -665.6993408, 1155.6306152, -2074.0468750, 2248.6599121
2: -804.7765503, 1667.8305664, -583.1625977, 1216.3449707, -2021.1214600, 2250.9931641
3: -1237.2823486, 1644.5600586, -896.4193726, 1198.0108643, -2435.2932129, 2540.9790039
4: -977.2471924, 1793.1127930, -706.7082520, 1305.4765625, -2282.7236328, 2499.8205566

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3600213, upper bound: 2027.3691735
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3598844, upper bound: 2027.3650575
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1010.0392456, 1753.9990234, -761.4674683, 1327.9113770, -2337.9506836, 2515.4665527
1: -915.3966064, 1577.9221191, -689.7806396, 1197.0191650, -2112.4157715, 2267.7019043
2: -802.1381226, 1662.4378662, -604.4384155, 1260.0015869, -2062.1396484, 2266.8762207
3: -1233.0527344, 1639.2479248, -928.6542969, 1241.0928955, -2474.1455078, 2567.9023438
4: -974.0643921, 1787.1630859, -732.5615234, 1352.4227295, -2326.4870605, 2519.7246094

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3720176, upper bound: 2027.3717392
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3722991, upper bound: 2027.3735698
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1098.0639648, 1907.0711670, -742.8050537, 1295.6810303, -2393.7448730, 2649.8759766
1: -994.7321167, 1716.6870117, -672.9194946, 1167.9333496, -2162.6655273, 2389.6064453
2: -871.3232422, 1808.3763428, -589.5203247, 1229.4130859, -2100.7363281, 2397.8967285
3: -1339.4243164, 1783.3439941, -906.2413330, 1210.8880615, -2550.3125000, 2689.5854492
4: -1058.0905762, 1943.1984863, -714.4996948, 1319.5310059, -2377.6213379, 2657.6982422

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3600852, upper bound: 2027.3690152
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3599384, upper bound: 2027.3635043
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1088.0648193, 1889.8679199, -765.4864502, 1334.8552246, -2422.9199219, 2655.3540039
1: -985.6697998, 1700.9976807, -693.4832153, 1203.2103271, -2188.8801270, 2394.4807129
2: -863.4324341, 1791.9133301, -607.6954956, 1266.6091309, -2130.0410156, 2399.6079102
3: -1327.2353516, 1767.0783691, -933.7526245, 1247.6253662, -2574.8608398, 2700.8308105
4: -1048.5830078, 1925.4462891, -736.5632935, 1359.5849609, -2408.1679688, 2662.0095215

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3722354, upper bound: 2027.3716660
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3723221, upper bound: 2027.3723220
time: 0.86 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.08 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3668108, upper bound: 2027.3679596
NS_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3675500, upper bound: 2027.3672824
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3668108, upper bound: 2027.3679596
NS_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3675500, upper bound: 2027.3672824
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3717880, upper bound: 2027.3729319
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3738252, upper bound: 2027.3750405
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3717880, upper bound: 2027.3729319
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3738252, upper bound: 2027.3750405
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3712660, upper bound: 2027.3740469
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3735050, upper bound: 2027.3775352
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3712660, upper bound: 2027.3740473
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3735050, upper bound: 2027.3775352
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3713500, upper bound: 2027.3750743
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3775352
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3713500, upper bound: 2027.3750743
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3775352
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3721949, upper bound: 2027.3734037
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3721800, upper bound: 2027.3729540
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3733707, upper bound: 2027.3741369
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3733119, upper bound: 2027.3733119
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3090327, upper bound: 2027.3411010
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3080446, upper bound: 2027.3363445
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3724056, upper bound: 2027.3719612
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3724331, upper bound: 2027.3722925
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3600213, upper bound: 2027.3691735
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3598844, upper bound: 2027.3650575
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3720176, upper bound: 2027.3717392
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3722991, upper bound: 2027.3735698
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3600852, upper bound: 2027.3690152
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3599384, upper bound: 2027.3635043
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3722354, upper bound: 2027.3716660
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.08
Output dim: 0, lower bound: -2027.3723221, upper bound: 2027.3723220

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -632.3406372, 1100.1535645, -657.1621704, 1143.5322266, -1775.8728027, 1757.3156738
1: -573.3193970, 990.2552490, -595.8187866, 1029.7995605, -1603.1188965, 1586.0734863
2: -502.4162903, 1044.0700684, -522.1062622, 1085.5240479, -1587.9403076, 1566.1762695
3: -772.5510864, 1027.5751953, -802.8892212, 1068.5133057, -1841.0644531, 1830.4643555
4: -609.5698853, 1122.0249023, -633.2908936, 1166.2967529, -1775.8666992, 1755.3156738

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3724360, upper bound: 2027.3737949
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3724360, upper bound: 2027.3737949
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -632.3406372, 1100.1535645, -1036.1872559, 1797.7734375, -2430.1137695, 2136.3408203
1: -573.3193970, 990.2552490, -939.1342773, 1617.2210693, -2190.5405273, 1929.3892822
2: -502.4162903, 1044.0700684, -823.0169678, 1704.4317627, -2206.8479004, 1867.0866699
3: -772.5510864, 1027.5751953, -1265.5798340, 1680.7491455, -2453.3002930, 2293.1550293
4: -609.5698853, 1122.0249023, -999.6131592, 1833.1767578, -2442.7460938, 2121.6376953

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 7

### Candidate
type: B, layer: 1, pos: 6

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 46

### Candidate
type: B, layer: 1, pos: 1

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3660378, upper bound: 2027.3672582
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3660378, upper bound: 2027.3672825
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -653.2119141, 1135.9114990, -659.0886841, 1147.0335693, -1800.2453613, 1795.0002441
1: -592.0430908, 1023.4752808, -597.4088745, 1033.3374023, -1625.3802490, 1620.8837891
2: -518.5236206, 1079.0343018, -523.3626709, 1089.2524414, -1607.7761230, 1602.3968506
3: -798.2245483, 1062.0762939, -805.2257080, 1072.0676270, -1870.2922363, 1867.3018799
4: -628.7502441, 1159.4313965, -634.6055908, 1170.2186279, -1798.9688721, 1794.0369873

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3734793, upper bound: 2027.3722213
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3756514, upper bound: 2027.3722007
time: 0.87 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -673.7100830, 1171.6151123, -654.9065552, 1140.0687256, -1813.7788086, 1826.5213623
1: -610.6709595, 1055.6702881, -593.6273804, 1026.9804688, -1637.6513672, 1649.2976074
2: -534.8591309, 1112.7470703, -520.0595093, 1082.4964600, -1617.3552246, 1632.8066406
3: -823.1069336, 1095.4066162, -800.1004028, 1065.4259033, -1888.5328369, 1895.5070801
4: -648.4570923, 1195.6646729, -630.6095581, 1162.8850098, -1811.3420410, 1826.2740479

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3738398, upper bound: 2027.3706264
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3738118, upper bound: 2027.3740872
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3764293, upper bound: 2027.3741555
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -653.2119141, 1135.9114990, -1029.5638428, 1787.0274658, -2440.2392578, 2165.4753418
1: -592.0430908, 1023.4752808, -932.8945312, 1607.8970947, -2199.9399414, 1956.3697510
2: -518.5236206, 1079.0343018, -817.4788818, 1694.3792725, -2212.9028320, 1896.5130615
3: -798.2245483, 1062.0762939, -1257.1473389, 1670.6820068, -2468.9062500, 2319.2236328
4: -628.7502441, 1159.4313965, -992.8379517, 1821.8317871, -2450.5815430, 2152.2690430

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3697092, upper bound: 2027.3537476
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3668374, upper bound: 2027.3536240
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -673.7100830, 1171.6151123, -1023.2596436, 1776.4040527, -2450.1142578, 2194.8747559
1: -610.6709595, 1055.6702881, -927.1995850, 1598.2301025, -2208.9001465, 1982.8698730
2: -534.8591309, 1112.7470703, -812.4987793, 1684.1469727, -2219.0061035, 1925.2458496
3: -823.1069336, 1095.4066162, -1249.3974609, 1660.5926514, -2483.6989746, 2344.8032227
4: -648.4570923, 1195.6646729, -986.8306885, 1810.7200928, -2459.1772461, 2182.4951172

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735017, upper bound: 2027.3680267
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3719646, upper bound: 2027.3679627
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -630.4646606, 1097.4631348, -706.0735474, 1229.1750488, -1859.6396484, 1803.5366211
1: -571.4086914, 988.5026855, -639.7210083, 1107.8887939, -1679.2974854, 1628.2235107
2: -500.5960083, 1041.7036133, -560.4818726, 1167.0822754, -1667.6779785, 1602.1855469
3: -770.1302490, 1025.4326172, -862.1397095, 1149.0280762, -1919.1583252, 1887.5721436
4: -606.9457397, 1119.0834961, -679.3002319, 1253.6759033, -1860.6213379, 1798.3837891

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3745418, upper bound: 2027.3726825
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3739508, upper bound: 2027.3727440
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -652.9299927, 1136.8294678, -702.3682251, 1222.9422607, -1875.8723145, 1839.1977539
1: -591.7598267, 1024.0260010, -636.3362427, 1102.2589111, -1694.0186768, 1660.3620605
2: -518.4367065, 1078.9100342, -557.5499878, 1161.1044922, -1679.5412598, 1636.4594727
3: -797.2200928, 1062.2528076, -857.5100098, 1143.1513672, -1940.3712158, 1919.7628174
4: -628.5731812, 1158.8642578, -675.8036499, 1247.1328125, -1875.7059326, 1834.6678467

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3756675, upper bound: 2027.3750542
time: 0.87 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3739508, upper bound: 2027.3751848
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -630.4646606, 1097.4631348, -1064.9305420, 1850.0799561, -2480.5444336, 2162.3933105
1: -571.4086914, 988.5026855, -964.8169556, 1665.4421387, -2236.8508301, 1953.3195801
2: -500.5960083, 1041.7036133, -845.3776855, 1753.9666748, -2254.5625000, 1887.0812988
3: -770.1302490, 1025.4326172, -1299.3647461, 1730.1207275, -2500.2507324, 2324.7971191
4: -606.9457397, 1119.0834961, -1026.3803711, 1884.9892578, -2491.9350586, 2145.4638672

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3712660, upper bound: 2027.3740209
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3712660, upper bound: 2027.3740468
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -652.9299927, 1136.8294678, -1059.6739502, 1841.1287842, -2494.0588379, 2196.5024414
1: -591.7598267, 1024.0260010, -960.0796509, 1657.3349609, -2249.0947266, 1984.1057129
2: -518.4367065, 1078.9100342, -841.2557373, 1745.3907471, -2263.8273926, 1920.1654053
3: -797.2200928, 1062.2528076, -1292.8624268, 1721.6717529, -2518.8918457, 2355.1145020
4: -628.5731812, 1158.8642578, -1021.4126587, 1875.6539307, -2504.2270508, 2180.2768555

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735050, upper bound: 2027.3774416
time: 0.93 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735050, upper bound: 2027.3775352
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -764.4501343, 1332.4968262, -719.8693237, 1253.3939209, -2017.8439941, 2052.3662109
1: -692.5327148, 1201.3239746, -652.2969971, 1129.6296387, -1822.1623535, 1853.6209717
2: -606.6989746, 1265.3319092, -571.5199585, 1190.0631104, -1796.7620850, 1836.8518066
3: -933.4954834, 1245.4147949, -879.0445557, 1171.6600342, -2105.1555176, 2124.4594727
4: -735.2786255, 1358.9248047, -692.8233643, 1278.3115234, -2013.5899658, 2051.7480469

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3772393, upper bound: 2027.3785425
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3793269, upper bound: 2027.3792856
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -774.7465820, 1349.9991455, -716.1292725, 1247.1048584, -2021.8514404, 2066.1284180
1: -701.9128418, 1217.1859131, -648.8818359, 1123.9848633, -1825.8977051, 1866.0676270
2: -615.1049194, 1282.2327881, -568.5772095, 1184.0639648, -1799.1689453, 1850.8100586
3: -946.0830688, 1261.9982910, -874.3483276, 1165.7573242, -2111.8403320, 2136.3466797
4: -745.4260254, 1377.2943115, -689.3175049, 1271.7340088, -2017.1600342, 2066.6118164

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3780005, upper bound: 2027.3788890
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3793594, upper bound: 2027.3793594
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -764.4501343, 1332.4968262, -1072.6385498, 1863.4578857, -2627.9079590, 2405.1352539
1: -692.5327148, 1201.3239746, -971.9332886, 1677.2270508, -2369.7597656, 2173.2568359
2: -606.6989746, 1265.3319092, -851.6856689, 1766.6157227, -2373.3146973, 2117.0175781
3: -933.4954834, 1245.4147949, -1309.1003418, 1742.4306641, -2675.9262695, 2554.5151367
4: -735.2786255, 1358.9248047, -1034.1259766, 1898.7485352, -2634.0263672, 2393.0507812

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3690129, upper bound: 2027.3702460
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3713500, upper bound: 2027.3748231
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3713500, upper bound: 2027.3749331
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -774.7465820, 1349.9991455, -1067.4766846, 1854.6662598, -2629.4128418, 2417.4758301
1: -701.9128418, 1217.1859131, -967.2687378, 1669.2611084, -2371.1738281, 2184.4545898
2: -615.1049194, 1282.2327881, -847.6275635, 1758.2008057, -2373.3056641, 2129.8598633
3: -946.0830688, 1261.9982910, -1302.7154541, 1734.1281738, -2680.2111816, 2564.7138672
4: -745.4260254, 1377.2943115, -1029.2423096, 1889.5974121, -2635.0234375, 2406.5366211

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3724157, upper bound: 2027.3738775
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3774259
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3774259
time: 1.09 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1047.4475098, 1818.0677490, -710.6008301, 1239.3007812, -2286.7482910, 2528.6684570
1: -949.2868042, 1635.8319092, -644.1735840, 1116.6424561, -2065.9289551, 2280.0053711
2: -832.0373535, 1723.8327637, -564.3621216, 1176.2108154, -2008.2481689, 2288.1948242
3: -1278.9399414, 1699.6213379, -867.6378174, 1157.8894043, -2436.8293457, 2567.2590332
4: -1010.4598389, 1853.6047363, -684.1473389, 1262.6745605, -2273.1340332, 2537.7519531

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3721949, upper bound: 2027.3733883
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3721949, upper bound: 2027.3733883
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1149.1010742, 1991.8514404, -704.1076660, 1228.0316162, -2377.1323242, 2695.9589844
1: -1041.8197021, 1791.3498535, -638.2637329, 1106.4674072, -2148.2868652, 2429.6132812
2: -913.0755615, 1888.8364258, -559.1795654, 1165.5052490, -2078.5808105, 2448.0156250
3: -1404.5906982, 1862.3532715, -859.6981812, 1147.3255615, -2551.9162598, 2722.0512695
4: -1109.2833252, 2032.5026855, -677.8687134, 1251.1539307, -2360.4372559, 2710.3708496

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3706135, upper bound: 2027.3682441
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3387982, upper bound: 2027.3595939
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1041.2786865, 1807.6528320, -726.2852173, 1266.7774658, -2308.0561523, 2533.9379883
1: -943.7000732, 1626.3380127, -658.3107300, 1141.3024902, -2085.0024414, 2284.6486816
2: -827.1665039, 1713.7911377, -576.7752686, 1202.0487061, -2029.2150879, 2290.5664062
3: -1271.3260498, 1689.6936035, -886.4171753, 1183.4968262, -2454.8227539, 2576.1108398
4: -1004.5791626, 1842.6994629, -699.1778564, 1290.2973633, -2294.8764648, 2541.8771973

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3714213, upper bound: 2027.3722726
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3713723, upper bound: 2027.3722929
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1142.2869873, 1980.2700195, -722.6707153, 1260.4439697, -2402.7309570, 2702.9406738
1: -1035.6177979, 1780.8339844, -655.0480957, 1135.6269531, -2171.2443848, 2435.8818359
2: -907.6643677, 1877.6966553, -573.9024658, 1196.0566406, -2103.7209473, 2451.5981445
3: -1396.1171875, 1851.3781738, -882.0217285, 1177.6281738, -2573.7453613, 2733.3999023
4: -1102.7344971, 2020.4036865, -695.7004395, 1283.8673096, -2386.6018066, 2716.1037598

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3536651, upper bound: 2027.3707379
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3733089, upper bound: 2027.3733089
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1072.1127930, 1863.5816650, -725.1991577, 1264.9735107, -2337.0859375, 2588.7807617
1: -971.3614502, 1677.7556152, -657.2620239, 1139.7192383, -2111.0803223, 2335.0175781
2: -851.2288818, 1767.0328369, -575.8327026, 1200.3253174, -2051.5541992, 2342.8649902
3: -1308.1412354, 1742.7006836, -884.9867554, 1181.8120117, -2489.9531250, 2627.6875000
4: -1033.4342041, 1898.7952881, -698.0120239, 1288.3818359, -2321.8156738, 2596.8073730

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3529702, upper bound: 2027.3679516
time: 0.77 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3724056, upper bound: 2027.3719612
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1082.4658203, 1880.8170166, -732.5971680, 1277.7427979, -2360.2082520, 2613.4140625
1: -980.8331909, 1693.2000732, -664.0157471, 1151.1585693, -2131.9914551, 2357.2158203
2: -859.4739380, 1783.3037109, -581.7479248, 1212.4183350, -2071.8913574, 2365.0510254
3: -1320.8460693, 1758.9232178, -894.1437988, 1193.7611084, -2514.6064453, 2653.0668945
4: -1043.4980469, 1916.4682617, -705.1978149, 1301.4643555, -2344.9624023, 2621.6660156

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3529864, upper bound: 2027.3688007
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3724331, upper bound: 2027.3722925
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -991.4258423, 1721.4471436, -723.8552856, 1262.6760254, -2254.1015625, 2445.3024902
1: -898.5516357, 1548.7680664, -655.6906128, 1138.3952637, -2036.9467773, 2204.4584961
2: -787.4793701, 1631.8002930, -574.4089966, 1198.2044678, -1985.6835938, 2206.2092285
3: -1210.4458008, 1608.9711914, -882.9340820, 1180.0952148, -2390.5410156, 2491.9050293
4: -956.2733154, 1754.3914795, -696.0944214, 1285.9644775, -2242.2377930, 2450.4858398

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3557343, upper bound: 2027.3688722
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3557343, upper bound: 2027.3688722
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -990.2972412, 1719.7675781, -746.6582642, 1302.1412354, -2292.4384766, 2466.4252930
1: -897.5818481, 1547.0717773, -676.4171143, 1173.7712402, -2071.3530273, 2223.4887695
2: -786.5347900, 1629.9659424, -592.7315674, 1235.5949707, -2022.1297607, 2222.6975098
3: -1209.1960449, 1607.2868652, -910.8256836, 1216.9726562, -2426.1687012, 2518.1125488
4: -955.1652222, 1752.3652344, -718.3157959, 1326.3133545, -2281.4782715, 2470.6811523

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3704607, upper bound: 2027.3715930
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3719231, upper bound: 2027.3717312
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1009.0746460, 1752.3525391, -741.0775146, 1292.4302979, -2301.5046387, 2493.4301758
1: -914.5971680, 1575.8944092, -671.2669067, 1165.0642090, -2079.6611328, 2247.1611328
2: -801.9070435, 1660.8607178, -588.2558594, 1226.4152832, -2028.3222656, 2249.1166992
3: -1232.3769531, 1637.4122314, -903.8356323, 1207.9066162, -2440.2832031, 2541.2475586
4: -973.7940063, 1786.0035400, -712.9403076, 1316.3739014, -2290.1672363, 2498.9438477

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3712997, upper bound: 2027.3735009
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3722269, upper bound: 2027.3735545
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1075.4620361, 1867.5697021, -732.1059570, 1277.1333008, -2352.5949707, 2599.6757812
1: -974.2559204, 1681.1279297, -663.2324219, 1151.2669678, -2125.5224609, 2344.3601074
2: -853.4925537, 1770.7285156, -581.0486450, 1211.8686523, -2065.3613281, 2351.7770996
3: -1311.6806641, 1746.4310303, -893.1808472, 1193.5639648, -2505.2446289, 2639.6118164
4: -1036.4832764, 1902.8718262, -704.2282715, 1300.6481934, -2337.1308594, 2607.1000977

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 20

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 38

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3557808, upper bound: 2027.3685717
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3557808, upper bound: 2027.3685717
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1066.6009521, 1852.4511719, -751.4998169, 1310.5208740, -2377.1218262, 2603.9509277
1: -966.3398438, 1667.1826172, -680.8487549, 1181.2471924, -2147.5869141, 2348.0312500
2: -846.5314941, 1756.2054443, -596.6356812, 1243.5535889, -2090.0847168, 2352.8410645
3: -1301.3604736, 1732.1198730, -916.8918457, 1224.8436279, -2526.2041016, 2649.0114746
4: -1028.1101074, 1887.3621826, -723.1247559, 1334.9138184, -2363.0239258, 2610.4868164

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3711829, upper bound: 2027.3715893
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3721126, upper bound: 2027.3716555
time: 0.83 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1080.5026855, 1875.7027588, -744.2938843, 1298.0018311, -2378.5043945, 2619.9965820
1: -978.8432617, 1687.5168457, -674.2517700, 1170.0156250, -2148.8583984, 2361.7685547
2: -857.9733276, 1778.0325928, -590.9014893, 1231.7189941, -2089.6921387, 2368.9335938
3: -1318.1914062, 1753.5968018, -907.9783325, 1213.1566162, -2531.3476562, 2661.5747070
4: -1042.1343994, 1911.3634033, -716.1989136, 1322.1292725, -2364.2634277, 2627.5622559

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 28

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3713038, upper bound: 2027.3722105
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3722335, upper bound: 2027.3722335
time: 0.81 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.53 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3724360, upper bound: 2027.3737949
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3724360, upper bound: 2027.3737949
NS_A1_B1_A1_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3660378, upper bound: 2027.3672582
NS_A1_B1_A1_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3660378, upper bound: 2027.3672825
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3734793, upper bound: 2027.3722213
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3756514, upper bound: 2027.3722007
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3738118, upper bound: 2027.3740872
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3764293, upper bound: 2027.3741555
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3697092, upper bound: 2027.3537476
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3668374, upper bound: 2027.3536240
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3735017, upper bound: 2027.3680267
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3719646, upper bound: 2027.3679627
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3745418, upper bound: 2027.3726825
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3739508, upper bound: 2027.3727440
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3756675, upper bound: 2027.3750542
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3739508, upper bound: 2027.3751848
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3712660, upper bound: 2027.3740209
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3712660, upper bound: 2027.3740468
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3735050, upper bound: 2027.3774416
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3735050, upper bound: 2027.3775352
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3772393, upper bound: 2027.3785425
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3793269, upper bound: 2027.3792856
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3780005, upper bound: 2027.3788890
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3793594, upper bound: 2027.3793594
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3713500, upper bound: 2027.3748231
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3713500, upper bound: 2027.3749331
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3774259
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3774259
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3721949, upper bound: 2027.3733883
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3721949, upper bound: 2027.3733883
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3706135, upper bound: 2027.3682441
NS_A2_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3387982, upper bound: 2027.3595939
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3714213, upper bound: 2027.3722726
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3713723, upper bound: 2027.3722929
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3536651, upper bound: 2027.3707379
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3733089, upper bound: 2027.3733089
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3529702, upper bound: 2027.3679516
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3724056, upper bound: 2027.3719612
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3529864, upper bound: 2027.3688007
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3724331, upper bound: 2027.3722925
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3557343, upper bound: 2027.3688722
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3557343, upper bound: 2027.3688722
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3704607, upper bound: 2027.3715930
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3719231, upper bound: 2027.3717312
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3712997, upper bound: 2027.3735009
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3722269, upper bound: 2027.3735545
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3557808, upper bound: 2027.3685717
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3557808, upper bound: 2027.3685717
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3711829, upper bound: 2027.3715893
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3721126, upper bound: 2027.3716555
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3713038, upper bound: 2027.3722105
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.53
Output dim: 0, lower bound: -2027.3722335, upper bound: 2027.3722335

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -632.3406372, 1100.1535645, -621.6230469, 1081.5000000, -1713.8405762, 1721.7763672
1: -573.3193970, 990.2552490, -563.5673828, 973.5086670, -1546.8281250, 1553.8223877
2: -502.4162903, 1044.0700684, -493.8988342, 1026.3818359, -1528.7979736, 1537.9688721
3: -772.5510864, 1027.5751953, -759.4181519, 1010.1683350, -1782.7194824, 1786.9934082
4: -609.5698853, 1122.0249023, -599.2044067, 1102.9953613, -1712.5651855, 1721.2291260

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -632.3406372, 1100.1535645, -657.8784180, 1144.4157715, -1776.7563477, 1758.0319824
1: -573.3193970, 990.2552490, -596.2476807, 1031.1597900, -1604.4792480, 1586.5025635
2: -502.4162903, 1044.0700684, -522.2158813, 1086.9519043, -1589.3681641, 1566.2858887
3: -772.5510864, 1027.5751953, -803.7364502, 1069.9591064, -1842.5102539, 1831.3116455
4: -609.5698853, 1122.0249023, -633.2264404, 1167.7707520, -1777.3405762, 1755.2513428

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 46

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 1

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 8

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -641.0350952, 1114.7719727, -639.4840698, 1113.0440674, -1754.0791016, 1754.2561035
1: -580.9539795, 1004.5219116, -579.5682983, 1002.6626587, -1583.6163330, 1584.0900879
2: -508.8246460, 1059.0415039, -507.7805481, 1056.9288330, -1565.7532959, 1566.8220215
3: -783.3178711, 1042.3551025, -781.0605469, 1040.1846924, -1823.5025635, 1823.4156494
4: -616.9671631, 1137.9003906, -615.7150879, 1135.3979492, -1752.3651123, 1753.6153564

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3734793, upper bound: 2027.3722213
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3734793, upper bound: 2027.3722213
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -647.7576904, 1126.4631348, -648.1401367, 1128.0621338, -1775.8197021, 1774.6032715
1: -587.1111450, 1014.9917603, -587.5030518, 1016.3104858, -1603.4215088, 1602.4948730
2: -514.2063599, 1070.1109619, -514.6884766, 1071.3348389, -1585.5412598, 1584.7994385
3: -791.6486206, 1053.2631836, -792.0111084, 1054.3626709, -1846.0112305, 1845.2742920
4: -623.5043945, 1149.8610840, -624.0541992, 1151.0012207, -1774.5056152, 1773.9151611

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3756514, upper bound: 2027.3722007
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3756514, upper bound: 2027.3722007
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -660.8428345, 1149.2827148, -634.5637207, 1104.8304443, -1765.6732178, 1783.8463135
1: -598.9761353, 1035.6274414, -575.1615601, 995.1799927, -1594.1561279, 1610.7886963
2: -524.6356201, 1091.6241455, -503.9134827, 1049.0074463, -1573.6429443, 1595.5375977
3: -807.3845825, 1074.5499268, -775.1390991, 1032.3620605, -1839.7463379, 1849.6889648
4: -636.0332642, 1172.9542236, -611.0070190, 1126.8397217, -1762.8725586, 1783.9611816

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3738118, upper bound: 2027.3740872
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3738118, upper bound: 2027.3740872
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -668.8366089, 1163.1899414, -644.1593628, 1121.4470215, -1790.2830811, 1807.3492432
1: -606.2630615, 1048.1026611, -583.9067993, 1010.2696533, -1616.5325928, 1632.0095215
2: -530.9963379, 1104.7883301, -511.5469360, 1064.9149170, -1595.9112549, 1616.3352051
3: -817.2283936, 1087.5446777, -787.1410522, 1048.0512695, -1865.2795410, 1874.6857910
4: -643.7646484, 1187.1185303, -620.2511597, 1144.0305176, -1787.7951660, 1807.3696289

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3764293, upper bound: 2027.3741555
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3764293, upper bound: 2027.3741555
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -643.2130127, 1118.5573730, -1008.0104980, 1749.6053467, -2392.8183594, 2126.5678711
1: -582.9984741, 1007.8735352, -913.3523560, 1574.2596436, -2157.2580566, 1921.2258301
2: -510.6425476, 1062.6367188, -800.4653320, 1658.9407959, -2169.5832520, 1863.1018066
3: -786.0591431, 1045.8568115, -1230.7325439, 1635.6893311, -2421.7485352, 2276.5891113
4: -619.2015381, 1141.8155518, -972.2211304, 1783.7387695, -2402.9399414, 2114.0366211

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3697092, upper bound: 2027.3537476
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3697092, upper bound: 2027.3537476
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -662.9600830, 1152.9562988, -1001.2805176, 1738.2255859, -2401.1855469, 2154.2368164
1: -600.9412842, 1038.9028320, -907.2670288, 1563.9095459, -2164.8505859, 1946.1695557
2: -526.3723145, 1095.1279297, -795.1549683, 1648.0012207, -2174.3735352, 1890.2825928
3: -810.0325317, 1077.9770508, -1222.4620361, 1624.8815918, -2434.9133301, 2300.4389648
4: -638.1732178, 1176.7225342, -965.8053589, 1771.8798828, -2410.0527344, 2142.5278320

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3709188, upper bound: 2027.3629335
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3717538, upper bound: 2027.3669086
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -655.1528931, 1139.4676514, -1103.6016846, 1913.2441406, -2568.3969727, 2243.0690918
1: -593.8944702, 1026.7779541, -1000.3680420, 1720.6113281, -2314.5053711, 2027.1457520
2: -520.1905518, 1082.3425293, -876.6873779, 1814.2309570, -2334.4213867, 1959.0297852
3: -800.5640869, 1065.3701172, -1348.6835938, 1788.7423096, -2589.3063965, 2414.0534668
4: -630.6646729, 1162.9790039, -1065.1437988, 1951.9989014, -2582.6635742, 2228.1228027

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3687097, upper bound: 2027.3628414
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3699699, upper bound: 2027.3668388
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -590.1064453, 1026.8880615, -617.4848633, 1074.7083740, -1664.8148193, 1644.3729248
1: -534.8166504, 924.6816406, -559.1920166, 967.9524536, -1502.7690430, 1483.8735352
2: -468.5383606, 974.6034546, -489.9613037, 1019.7875977, -1488.3259277, 1464.5646973
3: -720.9735718, 959.3740845, -753.5136719, 1004.1732178, -1725.1467285, 1712.8873291
4: -568.2500000, 1047.0975342, -594.2718506, 1095.5205078, -1663.7703857, 1641.3693848

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3724908, upper bound: 2027.3714690
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3733969, upper bound: 2027.3713987
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3733164, upper bound: 2027.3616391
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -605.0900269, 1052.9703369, -669.8882446, 1164.8796387, -1769.9697266, 1722.8585205
1: -548.3317871, 948.5920410, -606.9748535, 1050.3024902, -1598.6342773, 1555.5668945
2: -480.2997131, 999.7450562, -531.7106934, 1106.8588867, -1587.1584473, 1531.4555664
3: -739.1953735, 984.0651245, -818.5881958, 1089.5993652, -1828.7945557, 1802.6533203
4: -582.3256226, 1074.0484619, -644.3156738, 1189.4989014, -1771.8244629, 1718.3641357

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3727285, upper bound: 2027.3714276
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3722892, upper bound: 2027.3616785
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -611.3530884, 1064.0278320, -613.6735229, 1068.2869873, -1679.6401367, 1677.7012939
1: -554.0650024, 958.1349487, -555.7319336, 962.1814575, -1516.2464600, 1513.8668213
2: -485.4529724, 1009.6205444, -486.9600220, 1013.6679688, -1499.1208496, 1496.5803223
3: -746.5317993, 994.0740967, -748.7738647, 998.1337891, -1744.6655273, 1742.8479004
4: -588.7681274, 1084.5659180, -590.6658936, 1088.8292236, -1677.5974121, 1675.2318115

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3745039, upper bound: 2027.3739442
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3742411, upper bound: 2027.3722348
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3747549, upper bound: 2027.3740292
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -627.1057129, 1091.5065918, -665.7064209, 1157.6378174, -1784.7435303, 1757.2128906
1: -568.2935181, 983.3732300, -603.1401978, 1043.7618408, -1612.0554199, 1586.5133057
2: -497.8101807, 1036.1658936, -528.3850708, 1099.9696045, -1597.7797852, 1564.5507812
3: -765.7276001, 1020.1083984, -813.4062500, 1082.8261719, -1848.5537109, 1833.5146484
4: -603.5418091, 1113.0072021, -640.3180542, 1182.0428467, -1785.5847168, 1753.3251953

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3708114, upper bound: 2027.3732071
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3741079, upper bound: 2027.3722921
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3741641, upper bound: 2027.3741455
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -630.4646606, 1097.4631348, -1050.0513916, 1824.1568604, -2454.6215820, 2147.5146484
1: -571.4086914, 988.5026855, -951.2827148, 1642.0966797, -2213.5053711, 1939.7854004
2: -500.5960083, 1041.7036133, -833.5152588, 1729.3358154, -2229.9318848, 1875.2188721
3: -770.1302490, 1025.4326172, -1281.1373291, 1705.8959961, -2476.0261230, 2306.5698242
4: -606.9457397, 1119.0834961, -1011.9845581, 1858.5040283, -2465.4497070, 2131.0678711

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3456881, upper bound: 2027.3652011
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3472180, upper bound: 2027.3663364
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -630.4646606, 1097.4631348, -1141.5312500, 1983.5578613, -2614.0224609, 2238.9941406
1: -571.4086914, 988.5026855, -1033.9844971, 1786.0974121, -2357.5058594, 2022.4871826
2: -500.5960083, 1041.7036133, -905.9611206, 1881.5787354, -2382.1748047, 1947.6647949
3: -770.1302490, 1025.4326172, -1393.3041992, 1855.1500244, -2625.2802734, 2418.7365723
4: -606.9457397, 1119.0834961, -1099.9243164, 2022.0528564, -2628.9985352, 2219.0078125

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3456881, upper bound: 2027.3682595
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3472180, upper bound: 2027.3685226
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -652.9299927, 1136.8294678, -1044.2524414, 1814.2552490, -2467.1853027, 2181.0815430
1: -591.7598267, 1024.0260010, -946.0513306, 1633.1201172, -2224.8796387, 1970.0773926
2: -518.4367065, 1078.9100342, -828.9498901, 1719.8364258, -2238.2731934, 1907.8597412
3: -797.2200928, 1062.2528076, -1273.9710693, 1696.5447998, -2493.7648926, 2336.2236328
4: -628.5731812, 1158.8642578, -1006.4854736, 1848.1782227, -2476.7512207, 2165.3496094

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3716460, upper bound: 2027.3735287
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3717354, upper bound: 2027.3761264
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -652.9299927, 1136.8294678, -1132.3031006, 1967.5106201, -2620.4406738, 2269.1325684
1: -591.7598267, 1024.0260010, -1025.5992432, 1771.5914307, -2363.3513184, 2049.6252441
2: -518.4367065, 1078.9100342, -898.6665039, 1866.3360596, -2384.7724609, 1977.5765381
3: -797.2200928, 1062.2528076, -1381.9990234, 1840.1429443, -2637.3630371, 2444.2509766
4: -628.5731812, 1158.8642578, -1091.0999756, 2005.6406250, -2634.2133789, 2249.9643555

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 27

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3716460, upper bound: 2027.3736699
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3717354, upper bound: 2027.3762467
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -754.6017456, 1315.2504883, -692.1358643, 1204.8297119, -1959.4311523, 2007.3862305
1: -683.5692749, 1185.7855225, -627.0825195, 1086.0552979, -1769.6243896, 1812.8679199
2: -598.8371582, 1248.8923340, -549.3014526, 1144.0877686, -1742.9248047, 1798.1938477
3: -921.3808594, 1229.3029785, -845.2028198, 1126.4343262, -2047.8151855, 2074.5058594
4: -725.7508545, 1341.2369385, -665.7587280, 1229.0155029, -1954.7663574, 2006.9956055

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3770362, upper bound: 2027.3783333
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3770349, upper bound: 2027.3773461
time: 0.80 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -721.6770020, 1256.0625000, -669.7617798, 1164.2775879, -1885.9545898, 1925.8240967
1: -653.9176636, 1132.6717529, -606.9215698, 1049.3035889, -1703.2211914, 1739.5932617
2: -572.7940674, 1192.9379883, -531.7824097, 1105.3801270, -1678.1739502, 1724.7204590
3: -881.6049194, 1174.6044922, -817.6641846, 1088.9487305, -1970.5537109, 1992.2686768
4: -694.1622314, 1281.8358154, -645.0155029, 1187.7418213, -1881.9038086, 1926.8511963

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3772797, upper bound: 2027.3784960
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3772778, upper bound: 2027.3773475
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -763.0087280, 1329.4105225, -687.7650146, 1197.2392578, -1960.2480469, 2017.1752930
1: -691.2618408, 1198.6342773, -623.0673828, 1079.2375488, -1770.4992676, 1821.7016602
2: -605.7734375, 1262.6575928, -545.8162842, 1136.8804932, -1742.6538086, 1808.4738770
3: -931.7645874, 1242.7523193, -839.7423706, 1119.3721924, -2051.1367188, 2082.4946289
4: -734.1017456, 1356.2868652, -661.5610352, 1221.2160645, -1955.3177490, 2017.8479004

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3755995, upper bound: 2027.3760016
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3757780, upper bound: 2027.3759667
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -733.1546631, 1275.6435547, -665.6896973, 1157.4926758, -1890.6472168, 1941.3332520
1: -664.3262939, 1150.3822021, -603.2141113, 1043.1466064, -1707.4729004, 1753.5961914
2: -582.0901489, 1211.7728271, -528.5851440, 1098.8870850, -1680.9771729, 1740.3577881
3: -895.5561523, 1193.1010742, -812.6175537, 1082.5151367, -1978.0712891, 2005.7185059
4: -705.4087524, 1302.2565918, -641.1998901, 1180.6491699, -1886.0577393, 1943.4565430

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3757971, upper bound: 2027.3770008
time: 0.89 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3759687, upper bound: 2027.3759687
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -764.4501343, 1332.4968262, -1050.0513916, 1824.1568604, -2588.6069336, 2382.5483398
1: -692.5327148, 1201.3239746, -951.2827148, 1642.0966797, -2334.6291504, 2152.6064453
2: -606.6989746, 1265.3319092, -833.5152588, 1729.3358154, -2336.0344238, 2098.8471680
3: -933.4954834, 1245.4147949, -1281.1373291, 1705.8959961, -2639.3913574, 2526.5522461
4: -735.2786255, 1358.9248047, -1011.9845581, 1858.5040283, -2593.7827148, 2370.9094238

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3702966, upper bound: 2027.3702142
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3708117, upper bound: 2027.3730791
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -764.4501343, 1332.4968262, -1142.1846924, 1984.7091064, -2749.1591797, 2474.6816406
1: -692.5327148, 1201.3239746, -1034.5710449, 1787.1217041, -2479.6535645, 2235.8942871
2: -606.6989746, 1265.3319092, -906.4725952, 1882.6580811, -2489.3569336, 2171.8044434
3: -933.4954834, 1245.4147949, -1394.0979004, 1856.2080078, -2789.7036133, 2639.5124512
4: -735.2786255, 1358.9248047, -1100.5421143, 2023.2119141, -2758.4899902, 2459.4667969

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3702966, upper bound: 2027.3711852
time: 0.74 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3708117, upper bound: 2027.3731194
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -774.7465820, 1349.9991455, -1044.2524414, 1814.2552490, -2589.0017090, 2394.2514648
1: -701.9128418, 1217.1859131, -946.0513306, 1633.1201172, -2335.0327148, 2163.2373047
2: -615.1049194, 1282.2327881, -828.9498901, 1719.8364258, -2334.9414062, 2111.1823730
3: -946.0830688, 1261.9982910, -1273.9710693, 1696.5447998, -2642.6279297, 2535.9692383
4: -745.4260254, 1377.2943115, -1006.4854736, 1848.1782227, -2593.6042480, 2383.7797852

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3774258
time: 0.75 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735077, upper bound: 2027.3772800
time: 1.03 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -774.7465820, 1349.9991455, -1132.3085938, 1967.5207520, -2742.2670898, 2482.3076172
1: -701.9128418, 1217.1859131, -1025.6042480, 1771.5999756, -2473.5126953, 2242.7900391
2: -615.1049194, 1282.2327881, -898.6710205, 1866.3453369, -2481.4499512, 2180.9030762
3: -946.0830688, 1261.9982910, -1382.0058594, 1840.1520996, -2786.2351074, 2644.0039062
4: -745.4260254, 1377.2943115, -1091.1054688, 2005.6505127, -2751.0766602, 2468.3999023

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3774259
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3735077, upper bound: 2027.3772838
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1047.4475098, 1818.0677490, -665.9018555, 1158.9741211, -2206.4216309, 2483.9697266
1: -949.2868042, 1635.8319092, -603.6994019, 1044.0295410, -1993.3164062, 2239.5312500
2: -832.0373535, 1723.8327637, -529.0086670, 1100.5257568, -1932.5629883, 2252.8413086
3: -1278.9399414, 1699.6213379, -813.6708374, 1083.1439209, -2362.0839844, 2513.2915039
4: -1010.4598389, 1853.6047363, -641.5140381, 1182.4179688, -2192.8774414, 2495.1186523

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3699615, upper bound: 2027.3712193
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3721799, upper bound: 2027.3733883
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1047.4475098, 1818.0677490, -1047.0286865, 1817.1143799, -2864.5175781, 2865.0627441
1: -949.2868042, 1635.8319092, -948.8831177, 1634.8669434, -2584.1530762, 2584.7148438
2: -832.0373535, 1723.8327637, -831.6685791, 1722.8449707, -2554.8823242, 2555.5014648
3: -1278.9399414, 1699.6213379, -1278.5656738, 1698.7496338, -2977.6894531, 2978.1870117
4: -1010.4598389, 1853.6047363, -1010.0674438, 1852.6379395, -2863.0969238, 2863.6721191

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3699615, upper bound: 2027.3712193
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3721800, upper bound: 2027.3733883
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1133.3474121, 1964.8507080, -679.6466064, 1185.6104736, -2318.9572754, 2644.4973145
1: -1027.4932861, 1767.0747070, -616.0064087, 1068.1491699, -2095.6425781, 2383.0810547
2: -900.5347290, 1863.2652588, -539.7454834, 1125.1300049, -2025.6645508, 2403.0107422
3: -1385.3039551, 1837.0718994, -829.6154175, 1107.5224609, -2492.8264160, 2666.6872559
4: -1094.0356445, 2004.8831787, -654.3773804, 1207.6590576, -2301.6945801, 2659.2604980

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3697097, upper bound: 2027.3632572
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3630250, upper bound: 2027.3390861
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3630250, upper bound: 2027.3682441
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1024.5617676, 1778.8732910, -700.1724854, 1221.5434570, -2246.1052246, 2479.0458984
1: -928.5279541, 1600.4699707, -634.5899658, 1100.4564209, -2028.9841309, 2235.0600586
2: -813.8784180, 1686.5588379, -556.0239868, 1158.9893799, -1972.8677979, 2242.5827637
3: -1250.9589844, 1662.7634277, -854.4033203, 1141.0240479, -2391.9829102, 2517.1662598
4: -988.4085083, 1813.3547363, -674.0620728, 1243.9295654, -2232.3381348, 2487.4162598

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3701713, upper bound: 2027.3705959
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3703028, upper bound: 2027.3715466
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1037.6041260, 1801.2415771, -716.1167603, 1249.1055908, -2286.7097168, 2517.3583984
1: -940.3743896, 1620.5976562, -649.1021118, 1125.4506836, -2065.8249512, 2269.6997070
2: -824.2606812, 1707.7492676, -568.7155151, 1185.3669434, -2009.6275635, 2276.4648438
3: -1266.8714600, 1683.7528076, -874.1318359, 1167.0263672, -2433.8974609, 2557.8847656
4: -1001.0447998, 1836.2331543, -689.3741455, 1272.4160156, -2273.4606934, 2525.6069336

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3711890, upper bound: 2027.3722929
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3712087, upper bound: 2027.3722929
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1122.0378418, 1944.5128174, -683.2310791, 1191.4222412, -2313.4599609, 2627.7438965
1: -1017.3832397, 1748.7785645, -619.4608765, 1073.7026367, -2091.0859375, 2368.2395020
2: -891.7448120, 1843.9663086, -542.6924438, 1130.8116455, -2022.5561523, 2386.6584473
3: -1371.7147217, 1818.2144775, -834.0758057, 1113.4261475, -2485.1401367, 2652.2900391
4: -1083.4339600, 1984.4869385, -657.8370361, 1214.0113525, -2297.4453125, 2642.3234863

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3527973, upper bound: 2027.3688402
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3527936, upper bound: 2027.3691148
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1122.2829590, 1945.0239258, -699.2914429, 1219.2503662, -2341.5332031, 2644.3151855
1: -1017.4545898, 1749.3500977, -633.8045654, 1098.8588867, -2116.3134766, 2383.1545410
2: -891.7155762, 1844.3129883, -555.3483276, 1157.2415771, -2048.9570312, 2399.6613770
3: -1371.4587402, 1818.7958984, -853.3604126, 1139.5003662, -2510.9589844, 2672.1562500
4: -1083.3551025, 1984.5205078, -673.1239624, 1242.1781006, -2325.5332031, 2657.6442871

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3713068, upper bound: 2027.3713886
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3713163, upper bound: 2027.3713163
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1051.9260254, 1828.0532227, -684.6013184, 1193.9594727, -2245.8852539, 2512.6545410
1: -953.1497803, 1645.8748779, -620.6234131, 1075.9949951, -2029.1447754, 2266.4982910
2: -835.3346558, 1733.5002441, -543.6917725, 1133.1717529, -1968.5062256, 2277.1914062
3: -1283.6883545, 1709.6757812, -835.6074219, 1115.7310791, -2399.4194336, 2545.2832031
4: -1014.1336670, 1863.0191650, -658.9629517, 1216.4592285, -2230.5927734, 2521.9821777

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3300354, upper bound: 2027.3521603
time: 0.72 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3529521, upper bound: 2027.3668624
time: 1.00 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1051.7030029, 1827.6142578, -702.1661987, 1224.3917236, -2276.0947266, 2529.7802734
1: -952.7850342, 1645.6260986, -636.3483276, 1103.4807129, -2056.2653809, 2281.9736328
2: -834.9422607, 1733.0206299, -557.5581055, 1162.0819092, -1997.0240479, 2290.5786133
3: -1282.9744873, 1709.3979492, -856.8143921, 1144.2539062, -2427.2280273, 2566.2124023
4: -1013.6382446, 1862.2359619, -675.7796631, 1247.3251953, -2260.9633789, 2538.0156250

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3712260, upper bound: 2027.3718027
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3712349, upper bound: 2027.3699002
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1061.5836182, 1844.0766602, -692.8237305, 1208.1835938, -2269.7670898, 2536.9003906
1: -961.9783325, 1660.2366943, -628.1177368, 1088.7290039, -2050.7072754, 2288.3544922
2: -843.0157471, 1748.6302490, -550.2622681, 1146.6431885, -1989.6589355, 2298.8925781
3: -1295.5373535, 1724.7645264, -845.7684326, 1129.0187988, -2424.5561523, 2570.5329590
4: -1023.5103760, 1879.4659424, -667.0121460, 1231.0249023, -2254.5351562, 2546.4780273

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3527472, upper bound: 2027.3680209
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3528235, upper bound: 2027.3687478
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1062.0666504, 1844.8873291, -709.9286499, 1237.7725830, -2299.8388672, 2554.8159180
1: -962.2766113, 1661.0882568, -643.4168091, 1115.4782715, -2077.7548828, 2304.5051270
2: -843.2028809, 1749.3026123, -563.7490845, 1174.7487793, -2017.9515381, 2313.0517578
3: -1295.7070312, 1725.6468506, -866.3659668, 1156.7801514, -2452.4873047, 2592.0126953
4: -1023.7244873, 1879.9207764, -683.3061523, 1261.0097656, -2284.7343750, 2563.2270508

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 39

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3712496, upper bound: 2027.3721503
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3712294, upper bound: 2027.3704572
time: 1.17 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -991.4258423, 1721.4471436, -684.6937866, 1191.9752197, -2183.4006348, 2406.1408691
1: -898.5516357, 1548.7680664, -620.3526611, 1074.4143066, -1972.9656982, 2169.1206055
2: -787.4793701, 1631.8002930, -543.5633545, 1131.9248047, -1919.4040527, 2175.3637695
3: -1210.4458008, 1608.9711914, -836.1934204, 1114.2613525, -2324.7067871, 2445.1645508
4: -956.2733154, 1754.3914795, -658.7849121, 1215.9625244, -2172.2358398, 2413.1760254

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3556081, upper bound: 2027.3685583
time: 0.95 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3556081, upper bound: 2027.3688722
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -991.4258423, 1721.4471436, -1012.9096069, 1759.4472656, -2750.8730469, 2734.3566895
1: -898.5516357, 1548.7680664, -917.6912842, 1583.4260254, -2481.9775391, 2466.4592285
2: -787.4793701, 1631.8002930, -804.2721558, 1667.4580078, -2454.9367676, 2436.0725098
3: -1210.4458008, 1608.9711914, -1235.2932129, 1645.3493652, -2855.7949219, 2844.2644043
4: -956.2733154, 1754.3914795, -976.8331299, 1791.8242188, -2748.0976562, 2731.2246094

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3556080, upper bound: 2027.3685583
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3556081, upper bound: 2027.3688722
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -969.8276978, 1683.7957764, -704.9649048, 1229.0681152, -2198.8957520, 2388.7607422
1: -879.1032104, 1514.7489014, -638.7263184, 1108.2374268, -1987.3404541, 2153.4746094
2: -770.4077148, 1595.9625244, -559.6682129, 1166.4864502, -1936.8940430, 2155.6303711
3: -1184.4080811, 1573.7956543, -859.9302368, 1149.1057129, -2333.5136719, 2433.7253418
4: -935.6049805, 1716.0749512, -678.1487427, 1252.2867432, -2187.8916016, 2394.2236328

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3689962, upper bound: 2027.3714607
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3690196, upper bound: 2027.3677320
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -970.2092896, 1684.3817139, -722.4183960, 1259.5980225, -2229.8073730, 2406.8000488
1: -879.3286743, 1515.4425049, -654.3255005, 1135.7640381, -2015.0927734, 2169.7680664
2: -770.5330811, 1596.4636230, -573.3879395, 1195.4431152, -1965.9761963, 2169.8515625
3: -1184.4143066, 1574.5321045, -881.0276489, 1177.5422363, -2361.9558105, 2455.5593262
4: -935.7279053, 1716.3410645, -694.7993774, 1283.0611572, -2218.7890625, 2411.1398926

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3700812, upper bound: 2027.3701456
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3701046, upper bound: 2027.3688741
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -987.9959717, 1715.2770996, -698.9013672, 1218.4711914, -2206.4672852, 2414.1784668
1: -895.5979614, 1542.6417236, -633.1368408, 1098.7670898, -1994.3649902, 2175.7785645
2: -785.3169556, 1625.8391113, -554.8004761, 1156.4832764, -1941.8002930, 2180.6389160
3: -1206.9014893, 1602.9685059, -852.3469849, 1139.2581787, -2346.1596680, 2455.3151855
4: -953.6760254, 1748.6242676, -672.2645874, 1241.4644775, -2195.1401367, 2420.8889160

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3707493, upper bound: 2027.3734953
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3707186, upper bound: 2027.3727596
time: 0.80 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -989.5120850, 1717.8465576, -716.1317139, 1248.6772461, -2238.1894531, 2433.9780273
1: -896.8362427, 1545.0693359, -648.5513306, 1125.9597168, -2022.7957764, 2193.6206055
2: -786.3288574, 1628.2309570, -568.3707275, 1185.1258545, -1971.4547119, 2196.6015625
3: -1208.3050537, 1605.4735107, -873.2249146, 1167.3399658, -2375.6450195, 2478.6984863
4: -954.8545532, 1750.9515381, -688.7706299, 1271.9117432, -2226.7661133, 2439.7221680

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 12

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3706673, upper bound: 2027.3734976
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3706362, upper bound: 2027.3727515
time: 0.91 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1075.4620361, 1867.5697021, -698.6195068, 1216.4233398, -2291.8852539, 2566.1887207
1: -974.2559204, 1681.1279297, -633.0405273, 1096.3312988, -2070.5869141, 2314.1684570
2: -853.4925537, 1770.7285156, -554.7092896, 1155.1064453, -2008.5989990, 2325.4375000
3: -1311.6806641, 1746.4310303, -853.2615967, 1137.0948486, -2448.7753906, 2599.6923828
4: -1036.4832764, 1902.8718262, -672.4422607, 1240.8146973, -2277.2978516, 2575.3139648

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3556021, upper bound: 2027.3672903
time: 0.88 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3556021, upper bound: 2027.3685717
time: 0.94 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1075.4620361, 1867.5697021, -1050.3669434, 1824.6114502, -2900.0668945, 2917.9365234
1: -974.2559204, 1681.1279297, -951.7235718, 1642.2025146, -2616.4584961, 2632.8515625
2: -853.4925537, 1770.7285156, -834.0546875, 1729.7247314, -2583.2172852, 2604.7822266
3: -1311.6806641, 1746.4310303, -1281.9290771, 1706.1169434, -3017.7976074, 3028.3598633
4: -1036.4832764, 1902.8718262, -1012.8088989, 1859.1718750, -2895.6552734, 2915.6806641

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 6

### Candidate
type: A, layer: 1, pos: 7

### Candidate
type: A, layer: 1, pos: 12

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -2027.3556021, upper bound: 2027.3672903
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3556020, upper bound: 2027.3685717
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1046.1379395, 1816.2492676, -712.4538574, 1242.1478271, -2288.2856445, 2528.7026367
1: -947.8908081, 1634.6208496, -645.5570679, 1119.8829346, -2067.7731934, 2280.1779785
2: -830.4417114, 1721.9104004, -565.6754150, 1178.8634033, -2009.3051758, 2287.5859375
3: -1276.5904541, 1698.4716797, -869.2424316, 1161.2778320, -2437.8681641, 2567.7136230
4: -1008.5964966, 1850.8704834, -685.6083374, 1265.6069336, -2274.2033691, 2536.4787598

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3694259, upper bound: 2027.3698373
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3694747, upper bound: 2027.3690183
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1045.3165283, 1815.1218262, -727.5207520, 1268.4121094, -2313.7280273, 2542.6425781
1: -946.9968872, 1633.8510742, -658.9580078, 1143.7493896, -2090.7460938, 2292.8090820
2: -829.5535278, 1720.9228516, -577.4606934, 1203.9074707, -2033.4609375, 2298.3835449
3: -1275.1153564, 1697.5692139, -887.2945557, 1185.9291992, -2461.0441895, 2584.8625488
4: -1007.4772949, 1849.3592529, -699.8108521, 1292.1370850, -2299.6142578, 2549.1699219

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3694849, upper bound: 2027.3715486
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3694365, upper bound: 2027.3677754
time: 1.12 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1059.4240723, 1838.5390625, -704.2712402, 1227.8878174, -2287.3120117, 2542.8103027
1: -959.8546143, 1654.0933838, -638.0780029, 1107.1170654, -2066.9716797, 2292.1706543
2: -841.4137573, 1742.8988037, -559.1662598, 1165.3828125, -2006.7965088, 2302.0649414
3: -1292.7767334, 1719.0479736, -859.1477051, 1148.0062256, -2440.7827148, 2578.1953125
4: -1022.0445557, 1873.9416504, -677.7053833, 1251.0776367, -2273.1220703, 2551.6469727

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3695843, upper bound: 2027.3700542
time: 0.84 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3695925, upper bound: 2027.3704088
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1059.5594482, 1838.8763428, -720.3874512, 1256.0457764, -2315.6052246, 2559.2636719
1: -959.8140259, 1654.6130371, -652.4270020, 1132.6531982, -2092.4670410, 2307.0397949
2: -841.2587280, 1743.1711426, -571.7917480, 1192.2147217, -2033.4732666, 2314.9628906
3: -1292.3959961, 1719.5081787, -878.4774170, 1174.3739014, -2466.7695312, 2597.9851074
4: -1021.8259277, 1873.8649902, -692.9584351, 1279.5096436, -2301.3352051, 2566.8234863

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 28

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3704189, upper bound: 2027.3700664
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -2027.3704236, upper bound: 2027.3704236
time: 0.85 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.92 seconds
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3734793, upper bound: 2027.3722213
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3734793, upper bound: 2027.3722213
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3756514, upper bound: 2027.3722007
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3756514, upper bound: 2027.3722007
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3738118, upper bound: 2027.3740872
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3738118, upper bound: 2027.3740872
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3764293, upper bound: 2027.3741555
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3764293, upper bound: 2027.3741555
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3697092, upper bound: 2027.3537476
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3697092, upper bound: 2027.3537476
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3709188, upper bound: 2027.3629335
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3717538, upper bound: 2027.3669086
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3687097, upper bound: 2027.3628414
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3699699, upper bound: 2027.3668388
NS_A1_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3733969, upper bound: 2027.3713987
NS_A1_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3733164, upper bound: 2027.3616391
NS_A1_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3727285, upper bound: 2027.3714276
NS_A1_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3722892, upper bound: 2027.3616785
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3742411, upper bound: 2027.3722348
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3747549, upper bound: 2027.3740292
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3741079, upper bound: 2027.3722921
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3741641, upper bound: 2027.3741455
NS_A1_B2_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3456881, upper bound: 2027.3652011
NS_A1_B2_A1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3472180, upper bound: 2027.3663364
NS_A1_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3456881, upper bound: 2027.3682595
NS_A1_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3472180, upper bound: 2027.3685226
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3716460, upper bound: 2027.3735287
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3717354, upper bound: 2027.3761264
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3716460, upper bound: 2027.3736699
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3717354, upper bound: 2027.3762467
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3770362, upper bound: 2027.3783333
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3770349, upper bound: 2027.3773461
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3772797, upper bound: 2027.3784960
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3772778, upper bound: 2027.3773475
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3755995, upper bound: 2027.3760016
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3757780, upper bound: 2027.3759667
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3757971, upper bound: 2027.3770008
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3759687, upper bound: 2027.3759687
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3702966, upper bound: 2027.3702142
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3708117, upper bound: 2027.3730791
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3702966, upper bound: 2027.3711852
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3708117, upper bound: 2027.3731194
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3774258
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3735077, upper bound: 2027.3772800
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3735166, upper bound: 2027.3774259
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3735077, upper bound: 2027.3772838
NS_A2_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3699615, upper bound: 2027.3712193
NS_A2_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3721799, upper bound: 2027.3733883
NS_A2_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3699615, upper bound: 2027.3712193
NS_A2_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3721800, upper bound: 2027.3733883
NS_A2_B1_A1_B1_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3630250, upper bound: 2027.3390861
NS_A2_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3630250, upper bound: 2027.3682441
NS_A2_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3701713, upper bound: 2027.3705959
NS_A2_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3703028, upper bound: 2027.3715466
NS_A2_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3711890, upper bound: 2027.3722929
NS_A2_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3712087, upper bound: 2027.3722929
NS_A2_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3527973, upper bound: 2027.3688402
NS_A2_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3527936, upper bound: 2027.3691148
NS_A2_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3713068, upper bound: 2027.3713886
NS_A2_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3713163, upper bound: 2027.3713163
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3300354, upper bound: 2027.3521603
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3529521, upper bound: 2027.3668624
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3712260, upper bound: 2027.3718027
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3712349, upper bound: 2027.3699002
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3527472, upper bound: 2027.3680209
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3528235, upper bound: 2027.3687478
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3712496, upper bound: 2027.3721503
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3712294, upper bound: 2027.3704572
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3556081, upper bound: 2027.3685583
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3556081, upper bound: 2027.3688722
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3556080, upper bound: 2027.3685583
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3556081, upper bound: 2027.3688722
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3689962, upper bound: 2027.3714607
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3690196, upper bound: 2027.3677320
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3700812, upper bound: 2027.3701456
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3701046, upper bound: 2027.3688741
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3707493, upper bound: 2027.3734953
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3707186, upper bound: 2027.3727596
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3706673, upper bound: 2027.3734976
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3706362, upper bound: 2027.3727515
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.VERIFIED, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3556021, upper bound: 2027.3672903
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3556021, upper bound: 2027.3685717
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3556021, upper bound: 2027.3672903
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3556020, upper bound: 2027.3685717
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3694259, upper bound: 2027.3698373
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3694747, upper bound: 2027.3690183
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3694849, upper bound: 2027.3715486
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3694365, upper bound: 2027.3677754
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3695843, upper bound: 2027.3700542
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3695925, upper bound: 2027.3704088
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3704189, upper bound: 2027.3700664
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.92
Output dim: 0, lower bound: -2027.3704236, upper bound: 2027.3704236

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -626.6144409, 1089.7703857, -639.4840698, 1113.0440674, -1739.6584473, 1729.2543945
1: -567.8265991, 981.9946289, -579.5682983, 1002.6626587, -1570.4887695, 1561.5627441
2: -497.3749084, 1035.3038330, -507.7805481, 1056.9288330, -1554.3035889, 1543.0843506
3: -765.6909790, 1018.9365845, -781.0605469, 1040.1846924, -1805.8753662, 1799.9970703
4: -603.0897217, 1112.3686523, -615.7150879, 1135.3979492, -1738.4876709, 1728.0836182

Time for backsubstitution: 1.68 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.69 + 417.68 = 421.37 seconds
