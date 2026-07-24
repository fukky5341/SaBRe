## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 1)
Time budget: 420 seconds
Split limit: 100
Threshold: 1379.30539580811


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-383.7346802, 1157.7136230, -383.7346802, 1157.7136230, -1541.4482422, 1541.4482422)
1: (-296.8671875, 769.2732544, -296.8671875, 769.2732544, -1066.1403809, 1066.1403809)
2: (-321.1560059, 731.4843140, -321.1560059, 731.4843140, -1052.6401367, 1052.6402588)
3: (-322.8305969, 936.9838257, -322.8305969, 936.9838257, -1259.8143311, 1259.8143311)
4: (-468.6798096, 792.5546875, -468.6798096, 792.5546875, -1261.2344971, 1261.2344971)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.68 + 1.86 = 3.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1379.3191890, upper bound: 1379.3191890

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3155315, upper bound: 1379.3136277
time: 0.76 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3165739, upper bound: 1379.3165739
time: 0.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.85 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 0, lower bound: -1379.3155315, upper bound: 1379.3136277
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.85
Output dim: 0, lower bound: -1379.3165739, upper bound: 1379.3165739

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -364.7656860, 1098.9184570, -370.7371216, 1117.6075439, -1482.3730469, 1469.6555176
1: -281.9913025, 730.2640991, -286.7138367, 742.6233521, -1024.6146240, 1016.9779053
2: -305.3290100, 694.3013306, -310.3312378, 706.1213989, -1011.4504395, 1004.6325684
3: -306.6088562, 889.5310669, -311.7484436, 904.6099243, -1211.2185059, 1201.2795410
4: -445.1815491, 752.4251709, -452.6183777, 765.2084351, -1210.3900146, 1205.0435791

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3147474, upper bound: 1379.3120081
time: 0.65 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3148207, upper bound: 1379.3126082
time: 0.81 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -404.9994812, 1222.7425537, -364.3931274, 1096.1668701, -1501.1663818, 1587.1356201
1: -312.9198914, 811.0145264, -281.5192871, 728.5759277, -1041.4958496, 1092.5336914
2: -338.4754028, 772.8442383, -304.6644592, 692.3125610, -1030.7879639, 1077.5086670
3: -341.8264160, 987.5227661, -306.4843750, 887.7765503, -1229.6025391, 1294.0070801
4: -493.3276978, 837.6828613, -444.0772705, 750.2734375, -1243.6009521, 1281.7600098

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3165505, upper bound: 1379.3149554
time: 0.86 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3149629, upper bound: 1379.3149630
time: 0.85 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.39 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -1379.3147474, upper bound: 1379.3120081
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -1379.3148207, upper bound: 1379.3126082
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -1379.3165505, upper bound: 1379.3149554
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.39
Output dim: 0, lower bound: -1379.3149629, upper bound: 1379.3149630

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -343.3803711, 1033.0273438, -370.8230591, 1121.0501709, -1464.4305420, 1403.8503418
1: -265.4270325, 686.7274170, -286.9209290, 744.5747070, -1010.0017090, 973.6483154
2: -287.4098206, 652.5768433, -310.0360107, 708.5881348, -995.9979248, 962.6128540
3: -288.8171387, 836.7213745, -312.2916870, 906.4738159, -1195.2910156, 1149.0128174
4: -418.7992249, 707.3523560, -452.8341980, 767.6717529, -1186.4709473, 1160.1865234

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3147469, upper bound: 1379.3119013
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3141461, upper bound: 1379.3120083
time: 0.61 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -362.2474976, 1091.2445068, -364.7232361, 1099.2008057, -1461.4481201, 1455.9677734
1: -280.0533142, 725.2034912, -282.0696106, 730.4930420, -1010.5461426, 1007.2730713
2: -303.2677917, 689.4873657, -305.3915405, 694.5736084, -997.8414307, 994.8789062
3: -304.4851990, 883.3359985, -306.6480713, 889.7493286, -1194.2344971, 1189.9841309
4: -442.1446838, 747.2253418, -445.3333435, 752.7369385, -1194.8813477, 1192.5583496

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3023497, upper bound: 1379.3076413
time: 0.62 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3146798, upper bound: 1379.3118770
time: 0.66 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -387.3774109, 1169.1228027, -364.5863342, 1099.4588623, -1486.8363037, 1533.7091064
1: -299.2821350, 775.4158325, -281.6472778, 730.0833130, -1029.3654785, 1057.0628662
2: -323.8123169, 738.8110352, -304.5072327, 694.4509277, -1018.2631836, 1043.3182373
3: -327.0814514, 944.1807251, -306.9521179, 889.3479614, -1216.4294434, 1251.1328125
4: -471.9118652, 800.8711548, -444.4352417, 752.4620972, -1224.3740234, 1245.3062744

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.2832140, upper bound: 1379.2844724
time: 0.58 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3165440, upper bound: 1379.3148208
time: 0.73 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -401.3581543, 1211.7580566, -355.9848633, 1070.4558105, -1471.8139648, 1567.7429199
1: -310.1233215, 803.7376099, -275.0669556, 711.5828857, -1021.7061768, 1078.8045654
2: -335.4910889, 765.9282837, -297.8073425, 676.0773315, -1011.5682373, 1063.7355957
3: -338.7765808, 978.6353760, -299.3667603, 867.0986328, -1205.8751221, 1278.0020752
4: -488.9408569, 830.1965942, -433.9051514, 732.7612305, -1221.7017822, 1264.1015625

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3015649, upper bound: 1379.2890255
time: 0.54 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3148309, upper bound: 1379.3148309
time: 0.65 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.89 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -1379.3147469, upper bound: 1379.3119013
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -1379.3141461, upper bound: 1379.3120083
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -1379.3023497, upper bound: 1379.3076413
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -1379.3146798, upper bound: 1379.3118770
NS_A2_B1_A1, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 0, lower bound: -1379.2832140, upper bound: 1379.2844724
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -1379.3165440, upper bound: 1379.3148208
NS_A2_B2_A1, status: Status.VERIFIED, split count: 3, time: 2.89
Output dim: 0, lower bound: -1379.3015649, upper bound: 1379.2890255
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.89
Output dim: 0, lower bound: -1379.3148309, upper bound: 1379.3148309

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -302.1871338, 907.9066772, -357.8436584, 1082.3721924, -1384.5593262, 1265.7503662
1: -233.5634155, 603.7675781, -277.1705322, 718.9602661, -952.5236816, 880.9379883
2: -253.4663239, 573.5617065, -299.6668701, 684.4280396, -937.8943481, 873.2285767
3: -254.1642609, 735.6691284, -301.5080566, 875.0877075, -1129.2519531, 1037.1771240
4: -368.9953918, 621.5944824, -437.4543762, 741.4973145, -1110.4925537, 1059.0487061

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3143161, upper bound: 1379.3116844
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3147458, upper bound: 1379.3112517
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -332.4532166, 998.3430786, -362.8207092, 1095.6212158, -1428.0744629, 1361.1638184
1: -256.8946228, 663.9689941, -280.7104797, 727.9212036, -984.8157959, 944.6794434
2: -278.2660828, 630.6408081, -303.3955383, 692.5449829, -970.8109741, 934.0363770
3: -279.6180420, 809.1980591, -305.5811462, 886.3109131, -1165.9289551, 1114.7791748
4: -405.1993103, 683.7473755, -442.8905945, 750.4309692, -1155.6301270, 1126.6376953

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3141438, upper bound: 1379.3112942
time: 0.57 seconds

## Relational analysis of NS_A1_B1_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3140506, upper bound: 1379.3102806
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3116247
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -310.4851379, 934.2199097, -351.2831116, 1057.6125488, -1368.0976562, 1285.5029297
1: -239.7614136, 620.2108154, -271.5414429, 702.8193970, -942.5808105, 891.7522583
2: -259.7158813, 589.3820801, -294.2740479, 668.1647339, -927.8805542, 883.6560669
3: -260.7913818, 755.5710449, -295.3033447, 856.1241455, -1116.9155273, 1050.8741455
4: -378.6607056, 638.7753906, -428.8897095, 724.1528320, -1102.8134766, 1067.6650391

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3023497, upper bound: 1379.3076413
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3023497, upper bound: 1379.3076413
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -356.0159607, 1071.0131836, -360.6499939, 1085.9241943, -1441.9400635, 1431.6630859
1: -275.0903015, 712.0476685, -278.8283997, 721.8690186, -996.9593506, 990.8759766
2: -298.0503540, 676.5735474, -301.9798584, 686.0833130, -984.1336670, 978.5534058
3: -299.0803223, 867.3601074, -303.1273193, 879.2833862, -1178.3634033, 1170.4874268
4: -434.3694458, 733.1704712, -440.2436523, 743.5192871, -1177.8886719, 1173.4140625

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3116956, upper bound: 1379.3118770
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3116956, upper bound: 1379.3118771
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -380.2947388, 1146.1655273, -361.2728577, 1088.5205078, -1468.8151855, 1507.4383545
1: -293.6888428, 760.5899658, -279.0093384, 722.9838257, -1016.6726074, 1039.5992432
2: -317.9203796, 724.2159424, -301.7511902, 687.3786011, -1005.2988892, 1025.9671631
3: -320.8564148, 926.2112427, -304.0759583, 880.7593994, -1201.6158447, 1230.2872314
4: -463.1995239, 784.9791870, -440.3020630, 744.7525635, -1207.9519043, 1225.2812500

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3160519, upper bound: 1379.3148174
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3164659, upper bound: 1379.3148168
time: 0.63 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -393.9883423, 1187.9453125, -352.6724548, 1059.6374512, -1453.6257324, 1540.6177979
1: -304.3153076, 788.3564453, -272.4298706, 704.5823364, -1008.8976440, 1060.7863770
2: -329.3720398, 750.7783203, -295.0256348, 669.1762085, -998.5480957, 1045.8037109
3: -332.3265686, 960.0053711, -296.5012207, 858.6036987, -1190.9299316, 1256.5062256
4: -479.8843079, 813.7049561, -429.7581787, 725.2364502, -1205.1207275, 1243.4630127

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3116917, upper bound: 1379.3146797
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3116917, upper bound: 1379.3148309
time: 0.67 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.18 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3143161, upper bound: 1379.3116844
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3147458, upper bound: 1379.3112517
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3140506, upper bound: 1379.3102806
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3116247
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3023497, upper bound: 1379.3076413
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3023497, upper bound: 1379.3076413
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3116956, upper bound: 1379.3118770
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3116956, upper bound: 1379.3118771
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3160519, upper bound: 1379.3148174
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3164659, upper bound: 1379.3148168
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3116917, upper bound: 1379.3146797
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.18
Output dim: 0, lower bound: -1379.3116917, upper bound: 1379.3148309

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -296.7948608, 891.5236816, -347.7329712, 1051.4648438, -1348.2596436, 1239.2565918
1: -229.4025726, 592.9439697, -269.3745422, 698.5856934, -927.9882812, 862.3184814
2: -249.0431976, 563.1705322, -291.4025879, 664.9428101, -913.9859619, 854.5731201
3: -249.6389771, 722.5109863, -293.0507812, 850.3164062, -1099.9552002, 1015.5617676
4: -362.5141907, 610.3225708, -425.2343140, 720.4125366, -1082.9267578, 1035.5567627

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3141418, upper bound: 1379.3116768
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3139927, upper bound: 1379.3109316
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -287.3410645, 863.6257935, -343.6138611, 1039.5637207, -1326.9047852, 1207.2396240
1: -222.2069397, 574.3812256, -266.2801819, 691.3096924, -913.5166016, 840.6613770
2: -241.3923187, 545.8719482, -288.5737305, 657.7869873, -899.1793213, 834.4456787
3: -241.5749969, 699.6693726, -289.6271057, 841.1295776, -1082.7045898, 989.2965088
4: -351.3379517, 591.5690918, -421.4349365, 712.7239990, -1064.0618896, 1013.0040283

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3143731, upper bound: 1379.3098957
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3147280, upper bound: 1379.3112510
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -330.5868835, 992.5703125, -360.8071899, 1088.6053467, -1419.1922607, 1353.3771973
1: -255.4482422, 660.1900635, -279.0191345, 723.4974976, -978.9456787, 939.2092285
2: -276.7248840, 627.0311890, -301.5596008, 688.0432129, -964.7680664, 928.5908203
3: -277.9686890, 804.5948486, -303.6295471, 881.0695801, -1159.0383301, 1108.2243652
4: -402.9314880, 679.8379517, -440.2216492, 745.5760498, -1148.5075684, 1120.0595703

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3102806
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3102807
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -331.6456299, 995.8340454, -359.7001953, 1085.9589844, -1417.6044922, 1355.5341797
1: -256.2672729, 662.3231201, -278.2844238, 721.5778809, -977.8450928, 940.6074829
2: -277.5983582, 629.0621338, -300.8114624, 686.4580688, -964.0563354, 929.8735962
3: -278.9382324, 807.2026978, -302.9001465, 878.6084595, -1157.5466309, 1110.1027832
4: -404.2117920, 682.0352173, -439.0953064, 743.8475342, -1148.0589600, 1121.1304932

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3116247
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3140212, upper bound: 1379.3116247
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -310.4851379, 934.2199097, -346.6556396, 1043.3620605, -1353.8471680, 1280.8754883
1: -239.7614136, 620.2108154, -267.9355774, 693.4337158, -933.1951294, 888.1463623
2: -259.7158813, 589.3820801, -290.4429932, 659.2019653, -918.9178467, 879.8249512
3: -260.7913818, 755.5710449, -291.3531189, 844.6375732, -1105.4289551, 1046.9239502
4: -378.6607056, 638.7753906, -423.2232971, 714.4540405, -1093.1147461, 1061.9986572

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3021703, upper bound: 1379.2946079
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3023497, upper bound: 1379.3076413
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -310.4851379, 934.2199097, -385.9034424, 1162.6387939, -1473.1239014, 1320.1232910
1: -239.7614136, 620.2108154, -297.7731323, 771.0481567, -1010.8095703, 917.9839478
2: -259.7158813, 589.3820801, -322.4308472, 734.7169189, -994.4328003, 911.8127441
3: -260.7913818, 755.5710449, -325.5332947, 938.8951416, -1199.6864014, 1081.1040039
4: -378.6607056, 638.7753906, -469.4078979, 796.4382324, -1175.0988770, 1108.1831055

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3021703, upper bound: 1379.2946079
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3021703, upper bound: 1379.3076413
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -356.0159607, 1071.0131836, -355.2422485, 1069.0446777, -1425.0605469, 1426.2553711
1: -275.0903015, 712.0476685, -274.5572510, 710.6967163, -985.7869873, 986.6047974
2: -298.0503540, 676.5735474, -297.4546814, 675.4082642, -973.4586182, 974.0281372
3: -299.0803223, 867.3601074, -298.4780884, 865.6748657, -1164.7550049, 1165.8381348
4: -434.3694458, 733.1704712, -433.5247803, 731.9683838, -1166.3378906, 1166.6953125

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3116859, upper bound: 1379.3118770
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3116953, upper bound: 1379.3117481
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -356.0159607, 1071.0131836, -392.1684570, 1182.8049316, -1538.8208008, 1463.1813965
1: -275.0903015, 712.0476685, -302.9432678, 784.8460693, -1059.9362793, 1014.9909668
2: -298.0503540, 676.5735474, -327.8916931, 747.6033325, -1045.6535645, 1004.4650879
3: -299.0803223, 867.3601074, -330.8746643, 955.6539917, -1254.7338867, 1198.2347412
4: -434.3694458, 733.1704712, -477.6978149, 810.3105469, -1244.6799316, 1210.8682861

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3116861, upper bound: 1379.3118770
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3116953, upper bound: 1379.3117481
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -374.6326599, 1128.7774658, -351.2169800, 1057.4670410, -1432.0997314, 1479.9943848
1: -289.3025513, 749.1099243, -271.2377319, 702.5428467, -991.8453369, 1020.3475952
2: -313.2562561, 713.2225952, -293.4298096, 667.7904053, -981.0465698, 1006.6524048
3: -316.1003723, 912.2589722, -295.6830139, 855.9700317, -1172.0701904, 1207.9416504
4: -456.3113098, 773.0745239, -427.9822998, 723.5911255, -1179.9022217, 1201.0566406

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3160455, upper bound: 1379.3148174
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3160455, upper bound: 1379.3148174
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -365.0974426, 1100.2054443, -345.3076172, 1039.5319824, -1404.6292725, 1445.5130615
1: -282.0014343, 730.1102295, -266.7171936, 691.0297241, -973.0311279, 996.8273926
2: -305.4411926, 695.3159180, -289.2313538, 656.7465820, -962.1877441, 984.5472412
3: -308.0374451, 888.9629517, -290.4682312, 841.8720703, -1149.9095459, 1179.4311523
4: -444.9064636, 753.6790161, -421.7737122, 711.7312012, -1156.6376953, 1175.4525146

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3164626, upper bound: 1379.3148168
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3164626, upper bound: 1379.3148168
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -393.9883423, 1187.9453125, -355.2422485, 1069.0446777, -1463.0329590, 1543.1875000
1: -304.3153076, 788.3564453, -274.5572510, 710.6967163, -1015.0120239, 1062.9136963
2: -329.3720398, 750.7783203, -297.4546814, 675.4082642, -1004.7802124, 1048.2327881
3: -332.3265686, 960.0053711, -298.4780884, 865.6748657, -1198.0009766, 1258.4832764
4: -479.8843079, 813.7049561, -433.5247803, 731.9683838, -1211.8526611, 1247.2297363

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112536, upper bound: 1379.3142782
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112411, upper bound: 1379.3146793
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -393.9883423, 1187.9453125, -392.1684570, 1182.8049316, -1576.7932129, 1580.1137695
1: -304.3153076, 788.3564453, -302.9432678, 784.8460693, -1089.1613770, 1091.2994385
2: -329.3720398, 750.7783203, -327.8916931, 747.6033325, -1076.9753418, 1078.6699219
3: -332.3265686, 960.0053711, -330.8746643, 955.6539917, -1287.9798584, 1290.8798828
4: -479.8843079, 813.7049561, -477.6978149, 810.3105469, -1290.1948242, 1291.4027100

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112537, upper bound: 1379.3144481
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112411, upper bound: 1379.3148295
time: 0.74 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.68 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3141418, upper bound: 1379.3116768
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3139927, upper bound: 1379.3109316
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3143731, upper bound: 1379.3098957
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3147280, upper bound: 1379.3112510
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3102806
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3102807
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3116247
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3140212, upper bound: 1379.3116247
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3021703, upper bound: 1379.2946079
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3023497, upper bound: 1379.3076413
NS_A1_B2_A1_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3021703, upper bound: 1379.2946079
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3021703, upper bound: 1379.3076413
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3116859, upper bound: 1379.3118770
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3116953, upper bound: 1379.3117481
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3116861, upper bound: 1379.3118770
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3116953, upper bound: 1379.3117481
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3160455, upper bound: 1379.3148174
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3160455, upper bound: 1379.3148174
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3164626, upper bound: 1379.3148168
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3164626, upper bound: 1379.3148168
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3112536, upper bound: 1379.3142782
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3112411, upper bound: 1379.3146793
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3112537, upper bound: 1379.3144481
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.68
Output dim: 0, lower bound: -1379.3112411, upper bound: 1379.3148295

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -293.6037903, 880.6179810, -345.8106995, 1045.5662842, -1339.1700439, 1226.4285889
1: -226.7963715, 586.0201416, -267.8973694, 694.7160645, -921.5123901, 853.9174194
2: -246.2453613, 556.2346191, -289.8256836, 661.2658691, -907.5112305, 846.0603027
3: -246.6212769, 714.2724609, -291.4048462, 845.5977173, -1092.2189941, 1005.6771851
4: -358.4029846, 602.8229980, -422.8966370, 716.4177856, -1074.8208008, 1025.7196045

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3137453, upper bound: 1379.3115546
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3141417, upper bound: 1379.3115571
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -293.7655640, 882.1503296, -346.9230042, 1048.9432373, -1342.7087402, 1229.0733643
1: -227.0610809, 586.7921143, -268.7433472, 696.9348145, -923.9959106, 855.5352783
2: -246.5464172, 557.2670898, -290.7300720, 663.3571167, -909.9035034, 847.9971313
3: -247.0891876, 715.0491943, -292.3533325, 848.3118896, -1095.4011230, 1007.4025269
4: -358.8244934, 603.9333496, -424.2443237, 718.6983032, -1077.5227051, 1028.1776123

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134905, upper bound: 1379.3107793
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3139926, upper bound: 1379.3109316
time: 1.06 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -271.1506958, 814.8688354, -337.8406067, 1022.7663574, -1293.9169922, 1152.7094727
1: -209.7653961, 542.2993774, -261.9158325, 680.0895996, -889.8549194, 804.2151489
2: -228.4668884, 514.9877319, -283.9839172, 647.2510376, -875.7178955, 798.9715576
3: -227.9690247, 660.3942261, -284.8920593, 827.2768555, -1055.2454834, 945.2862549
4: -332.4382019, 558.0135498, -414.6549377, 701.2420044, -1033.6801758, 972.6684570

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3136228, upper bound: 1379.3089231
time: 0.56 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3136228, upper bound: 1379.3098957
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -280.6477661, 842.6562500, -339.6844482, 1027.4509277, -1308.0986328, 1182.3406982
1: -216.9481964, 560.5359497, -263.2272034, 683.3466797, -900.2947998, 823.7630005
2: -235.8158264, 532.4113159, -285.3231201, 650.1116943, -885.9274902, 817.7343750
3: -235.8448486, 682.8944702, -286.1992493, 831.4475708, -1067.2924805, 969.0936890
4: -343.1062927, 576.9437256, -416.6725769, 704.3897705, -1047.4960938, 993.6162720

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3142146, upper bound: 1379.3086822
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3129426, upper bound: 1379.3087765
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -330.4986267, 991.4775391, -360.8071899, 1088.6053467, -1419.1040039, 1352.2846680
1: -255.2087250, 659.5679932, -279.0191345, 723.4974976, -978.7061768, 938.5871582
2: -276.4482117, 626.2039795, -301.5596008, 688.0432129, -964.4914551, 927.7635498
3: -277.6385803, 803.9555664, -303.6295471, 881.0695801, -1158.7081299, 1107.5850830
4: -402.5508728, 678.9116211, -440.2216492, 745.5760498, -1148.1269531, 1119.1333008

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -329.3869019, 988.8242798, -360.8071899, 1088.6053467, -1417.9921875, 1349.6314697
1: -254.5123901, 657.7211914, -279.0191345, 723.4974976, -978.0098877, 936.7402954
2: -275.7297668, 624.6491699, -301.5596008, 688.0432129, -963.7729492, 926.2087402
3: -277.0366516, 801.6262817, -303.6295471, 881.0695801, -1158.1062012, 1105.2558594
4: -401.4483948, 677.2485352, -440.2216492, 745.5760498, -1147.0244141, 1117.4702148

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -330.4986267, 991.4775391, -359.7001953, 1085.9589844, -1416.4576416, 1351.1777344
1: -255.2087250, 659.5679932, -278.2844238, 721.5778809, -976.7866211, 937.8524170
2: -276.4482117, 626.2039795, -300.8114624, 686.4580688, -962.9062500, 927.0153809
3: -277.6385803, 803.9555664, -302.9001465, 878.6084595, -1156.2469482, 1106.8555908
4: -402.5508728, 678.9116211, -439.0953064, 743.8475342, -1146.3983154, 1118.0068359

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3136453, upper bound: 1379.3114834
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3116247
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -329.3869019, 988.8242798, -359.7001953, 1085.9589844, -1415.3459473, 1348.5244141
1: -254.5123901, 657.7211914, -278.2844238, 721.5778809, -976.0902710, 936.0056152
2: -275.7297668, 624.6491699, -300.8114624, 686.4580688, -962.1878052, 925.4606323
3: -277.0366516, 801.6262817, -302.9001465, 878.6084595, -1155.6451416, 1104.5263672
4: -401.4483948, 677.2485352, -439.0953064, 743.8475342, -1145.2957764, 1116.3437500

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3136453, upper bound: 1379.3102831
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3097772
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -306.9619141, 923.1371460, -344.8086243, 1037.5025635, -1344.4642334, 1267.9456787
1: -237.0007782, 612.9458008, -266.4831238, 689.5938721, -926.5946655, 879.4289551
2: -256.7778015, 582.4044800, -288.9006653, 655.5108643, -912.2886353, 871.3051147
3: -257.7658997, 746.6986694, -289.7567444, 839.9523926, -1097.7181396, 1036.4554443
4: -374.2943420, 631.2342529, -420.9240112, 710.4666138, -1084.7608643, 1052.1582031

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3104207, upper bound: 1379.3106860
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3111647, upper bound: 1379.3107027
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -306.9619141, 923.1371460, -382.6343994, 1152.5656738, -1459.5270996, 1305.7713623
1: -237.0007782, 612.9458008, -295.2302551, 764.4048462, -1001.4056396, 908.1760254
2: -256.7778015, 582.4044800, -319.7149353, 728.4188232, -985.1965942, 902.1193848
3: -257.7658997, 746.6986694, -322.7628174, 930.7654419, -1188.5311279, 1069.4614258
4: -374.2943420, 631.2342529, -465.3663940, 789.5992432, -1163.8935547, 1096.6005859

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3007761, upper bound: 1379.3056551
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3007760, upper bound: 1379.3076413
time: 0.64 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -315.8409424, 949.2730103, -340.2694092, 1024.0485840, -1339.8895264, 1289.5421143
1: -244.0435181, 631.2462158, -263.1898804, 681.0020142, -925.0455322, 894.4360962
2: -264.9294128, 599.6348267, -285.4679565, 647.3905640, -912.3199463, 885.1027832
3: -265.3577271, 768.9592285, -286.0064697, 829.2517090, -1094.6093750, 1054.9656982
4: -385.7801819, 649.6176147, -415.8183899, 701.5579224, -1087.3381348, 1065.4360352

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3117325, upper bound: 1379.3112565
time: 0.58 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107830, upper bound: 1379.3112184
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -343.2005920, 1030.5622559, -347.5432739, 1044.5292969, -1387.7298584, 1378.1054688
1: -265.1484070, 685.5075073, -268.5626831, 694.6378784, -959.7862549, 954.0701904
2: -287.4013977, 651.0681152, -291.0312195, 659.9145508, -947.3159180, 942.0993042
3: -288.3349915, 835.2264404, -292.0010681, 846.2426758, -1134.5776367, 1127.2275391
4: -418.5112000, 705.7278442, -423.9243774, 715.3285522, -1133.8397217, 1129.6522217

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3117133, upper bound: 1379.3108031
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107655, upper bound: 1379.3107655
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -315.8409424, 949.2730103, -381.9171753, 1151.2006836, -1467.0416260, 1331.1901855
1: -244.0435181, 631.2462158, -295.0123901, 764.0362549, -1008.0797729, 926.2586060
2: -264.9294128, 599.6348267, -319.4242859, 727.9327393, -992.8621826, 919.0590820
3: -265.3577271, 768.9592285, -322.2438965, 930.0969849, -1195.4547119, 1091.2031250
4: -385.7801819, 649.6176147, -465.0171509, 788.9485474, -1174.7287598, 1114.6347656

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3142782, upper bound: 1379.3112537
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3146791, upper bound: 1379.3112411
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -343.2005920, 1030.5622559, -383.1773987, 1154.3476562, -1497.5482178, 1413.7390137
1: -265.1484070, 685.5075073, -295.9916992, 766.1774292, -1031.3258057, 981.4992065
2: -287.4013977, 651.0681152, -320.4701233, 729.6267090, -1017.0280762, 971.5382080
3: -288.3349915, 835.2264404, -323.3165588, 933.1298218, -1221.4647217, 1158.5429688
4: -418.5112000, 705.7278442, -466.6074829, 791.0352173, -1209.5462646, 1172.3353271

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3142085, upper bound: 1379.3108006
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3141847, upper bound: 1379.3107881
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -420.6780396, 1271.3027344, -351.2169800, 1057.4670410, -1478.1450195, 1622.5196533
1: -324.7078552, 842.7380981, -271.2377319, 702.5428467, -1027.2507324, 1113.9754639
2: -350.5513000, 802.8283691, -293.4298096, 667.7904053, -1018.3416138, 1096.2581787
3: -354.6867981, 1025.5581055, -295.6830139, 855.9700317, -1210.6567383, 1321.2409668
4: -511.8326111, 869.7347412, -427.9822998, 723.5911255, -1235.4235840, 1297.7170410

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3153901, upper bound: 1379.3146681
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3160455, upper bound: 1379.3148174
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -383.4564209, 1155.5578613, -351.2169800, 1057.4670410, -1440.9234619, 1506.7749023
1: -296.1496582, 766.9704590, -271.2377319, 702.5428467, -998.6925049, 1038.2078857
2: -320.6972961, 730.3368530, -293.4298096, 667.7904053, -988.4876709, 1023.7666626
3: -323.4373779, 933.9541626, -295.6830139, 855.9700317, -1179.4071045, 1229.6370850
4: -467.0663452, 791.5847168, -427.9822998, 723.5911255, -1190.6574707, 1219.5670166

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3153902, upper bound: 1379.3146681
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3160455, upper bound: 1379.3148174
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -411.7611084, 1244.2806396, -345.3076172, 1039.5319824, -1451.2930908, 1589.5882568
1: -317.8226013, 824.8917847, -266.7171936, 691.0297241, -1008.8522949, 1091.6090088
2: -343.1879272, 785.8825073, -289.2313538, 656.7465820, -999.9345093, 1075.1135254
3: -347.1008301, 1003.6473999, -290.4682312, 841.8720703, -1188.9729004, 1294.1156006
4: -501.0655823, 851.3718262, -421.7737122, 711.7312012, -1212.7963867, 1273.1452637

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3138914, upper bound: 1379.3147958
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3164626, upper bound: 1379.3148168
time: 1.17 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -375.1682739, 1130.6353760, -345.3076172, 1039.5319824, -1414.7001953, 1475.9429932
1: -289.7499084, 750.4230957, -266.7171936, 691.0297241, -980.7796631, 1017.1402588
2: -313.8466187, 714.7867432, -289.2313538, 656.7465820, -970.5932007, 1004.0180664
3: -316.3563232, 913.5909424, -290.4682312, 841.8720703, -1158.2283936, 1204.0592041
4: -457.0473633, 774.7087402, -421.7737122, 711.7312012, -1168.7783203, 1196.4821777

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3138914, upper bound: 1379.3147958
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3164626, upper bound: 1379.3148168
time: 1.02 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -384.0292358, 1157.2442627, -349.9929199, 1053.1639404, -1437.1928711, 1507.2370605
1: -296.5791321, 768.0635376, -270.5050964, 700.1893921, -996.7685547, 1038.5686035
2: -321.1578674, 731.3427734, -293.1550598, 665.3487549, -986.5065918, 1024.4978027
3: -323.9267578, 935.3665771, -294.0651245, 852.8620605, -1176.7888184, 1229.4315186
4: -467.7471924, 792.6695557, -427.2225952, 721.0427246, -1188.7899170, 1219.8920898

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112537, upper bound: 1379.3142782
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3108006, upper bound: 1379.3142085
time: 0.66 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -377.2628174, 1136.7121582, -339.3243103, 1021.5554199, -1398.8182373, 1476.0364990
1: -291.2961121, 754.8294678, -262.4060059, 679.1591797, -970.4552002, 1017.2353516
2: -315.9954224, 718.6023560, -284.5484314, 645.6505737, -961.6459961, 1003.1507568
3: -318.2994385, 919.2563477, -285.0006714, 827.0717773, -1145.3712158, 1204.2568359
4: -460.3416443, 778.9147339, -414.6447754, 699.7053833, -1160.0469971, 1193.5595703

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112411, upper bound: 1379.3146791
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107881, upper bound: 1379.3141847
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -384.0292358, 1157.2442627, -386.0328369, 1163.8631592, -1547.8923340, 1543.2770996
1: -296.5791321, 768.0635376, -298.1847229, 772.3383789, -1068.9174805, 1066.2482910
2: -321.1578674, 731.3427734, -322.8357544, 735.6196289, -1056.7774658, 1054.1784668
3: -323.9267578, 935.3665771, -325.6959839, 940.4636230, -1264.3900146, 1261.0625000
4: -467.7471924, 792.6695557, -470.2257080, 797.3485718, -1265.0955811, 1262.8951416

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3136824, upper bound: 1379.3144446
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3148175, upper bound: 1379.3144481
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -377.2628174, 1136.7121582, -377.6556091, 1138.6354980, -1515.8983154, 1514.3676758
1: -291.2961121, 754.8294678, -291.7052307, 755.5853271, -1046.8812256, 1046.5345459
2: -315.9954224, 718.6023560, -315.9047241, 719.8684692, -1035.8638916, 1034.5070801
3: -318.2994385, 919.2563477, -318.5345459, 919.8646240, -1238.1640625, 1237.7907715
4: -460.3416443, 778.9147339, -460.0788879, 780.2581177, -1240.5997314, 1238.9936523

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3137137, upper bound: 1379.3148228
time: 1.00 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3148130, upper bound: 1379.3148295
time: 0.66 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.22 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3137453, upper bound: 1379.3115546
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3141417, upper bound: 1379.3115571
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3134905, upper bound: 1379.3107793
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3139926, upper bound: 1379.3109316
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3136228, upper bound: 1379.3089231
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3136228, upper bound: 1379.3098957
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3142146, upper bound: 1379.3086822
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3129426, upper bound: 1379.3087765
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3136453, upper bound: 1379.3114834
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3116247
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3136453, upper bound: 1379.3102831
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3140213, upper bound: 1379.3097772
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3104207, upper bound: 1379.3106860
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3111647, upper bound: 1379.3107027
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3007761, upper bound: 1379.3056551
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3007760, upper bound: 1379.3076413
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3117325, upper bound: 1379.3112565
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3107830, upper bound: 1379.3112184
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3117133, upper bound: 1379.3108031
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3107655, upper bound: 1379.3107655
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3142782, upper bound: 1379.3112537
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3146791, upper bound: 1379.3112411
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3142085, upper bound: 1379.3108006
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3141847, upper bound: 1379.3107881
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3153901, upper bound: 1379.3146681
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3160455, upper bound: 1379.3148174
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3153902, upper bound: 1379.3146681
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3160455, upper bound: 1379.3148174
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3138914, upper bound: 1379.3147958
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3164626, upper bound: 1379.3148168
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3138914, upper bound: 1379.3147958
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3164626, upper bound: 1379.3148168
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3112537, upper bound: 1379.3142782
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3108006, upper bound: 1379.3142085
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3112411, upper bound: 1379.3146791
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3107881, upper bound: 1379.3141847
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3136824, upper bound: 1379.3144446
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3148175, upper bound: 1379.3144481
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3137137, upper bound: 1379.3148228
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.22
Output dim: 0, lower bound: -1379.3148130, upper bound: 1379.3148295

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -293.6037903, 880.6179810, -343.4040222, 1038.0156250, -1331.6192627, 1224.0219727
1: -226.7963715, 586.0201416, -266.0473328, 689.7423096, -916.5386353, 852.0674438
2: -246.2453613, 556.2346191, -287.8786011, 656.5622559, -902.8076172, 844.1132202
3: -246.6212769, 714.2724609, -289.3753357, 839.5045166, -1086.1257324, 1003.6477661
4: -358.4029846, 602.8229980, -419.9307861, 711.3106079, -1069.7136230, 1022.7537231

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3115545
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3115545
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -293.6037903, 880.6179810, -343.5857239, 1038.6292725, -1332.2330322, 1224.2033691
1: -226.7963715, 586.0201416, -266.1556396, 690.1550293, -916.9513550, 852.1757812
2: -246.2453613, 556.2346191, -287.9777527, 656.9071045, -903.1524658, 844.2124023
3: -246.6212769, 714.2724609, -289.4987488, 840.0092773, -1086.6306152, 1003.7712402
4: -358.4029846, 602.8229980, -420.1456299, 711.6771240, -1070.0800781, 1022.9686279

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134895, upper bound: 1379.3115570
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134895, upper bound: 1379.3115571
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -293.7655640, 882.1503296, -344.4926147, 1041.3255615, -1335.0910645, 1226.6429443
1: -227.0610809, 586.7921143, -266.8762512, 691.9092407, -918.9703369, 853.6683350
2: -246.5464172, 557.2670898, -288.7642517, 658.6127930, -905.1591797, 846.0313110
3: -247.0891876, 715.0491943, -290.3034668, 842.1582031, -1089.2474365, 1005.3526611
4: -358.8244934, 603.9333496, -421.2449951, 713.5477295, -1072.3721924, 1025.1783447

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -293.7655640, 882.1503296, -344.7047729, 1042.0303955, -1335.7958984, 1226.8551025
1: -227.0610809, 586.7921143, -267.0075378, 692.3906250, -919.4517212, 853.7996216
2: -246.5464172, 557.2670898, -288.8889771, 659.0137939, -905.5601807, 846.1560669
3: -247.0891876, 715.0491943, -290.4540405, 842.7434692, -1089.8326416, 1005.5032349
4: -358.8244934, 603.9333496, -421.5038147, 713.9739990, -1072.7983398, 1025.4371338

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -271.1506958, 814.8688354, -312.6641541, 946.2457275, -1217.3963623, 1127.5329590
1: -209.7653961, 542.2993774, -242.2364655, 629.2578125, -839.0231323, 784.5358276
2: -228.4668884, 514.9877319, -263.0335999, 598.6320190, -827.0988770, 778.0213623
3: -227.9690247, 660.3942261, -263.4041748, 765.5341797, -993.5031738, 923.7980957
4: -332.4382019, 558.0135498, -384.0086060, 648.4279785, -980.8661499, 942.0221558

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3081463, upper bound: 1379.3005450
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109062, upper bound: 1379.3049054
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -271.1506958, 814.8688354, -333.9268799, 1008.9319458, -1280.0826416, 1148.7956543
1: -209.7653961, 542.2993774, -258.7013245, 671.2027588, -880.9680786, 801.0006104
2: -228.4668884, 514.9877319, -280.5933838, 638.4251709, -866.8920898, 795.5811157
3: -227.9690247, 660.3942261, -281.4057922, 816.7433472, -1044.7122803, 941.7999878
4: -332.4382019, 558.0135498, -409.4958496, 691.8958130, -1024.3339844, 967.5093994

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3081463, upper bound: 1379.3012371
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109062, upper bound: 1379.3061660
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -278.9251099, 837.3625488, -337.2131958, 1019.0330811, -1297.9580078, 1174.5756836
1: -215.6141663, 557.0704346, -261.2059937, 678.0353394, -893.6495361, 818.2764282
2: -234.3984070, 529.1233521, -283.1549988, 644.7484131, -879.1468506, 812.2783203
3: -234.3268890, 678.6632690, -283.8608704, 825.1578979, -1059.4846191, 962.5241089
4: -341.0217590, 573.3677368, -413.5165100, 698.6024170, -1039.6240234, 986.8840942

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3129426, upper bound: 1379.3086822
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3129426, upper bound: 1379.3086822
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -279.7460632, 839.8630371, -336.4512329, 1017.4390869, -1297.1850586, 1176.3142090
1: -216.2500763, 558.6969604, -260.7100830, 676.7816162, -893.0316772, 819.4069824
2: -235.0686340, 530.6454468, -282.6421204, 643.8250122, -878.8936157, 813.2875366
3: -235.0869446, 680.6661377, -283.4180908, 823.4862671, -1058.5732422, 964.0841675
4: -341.9987488, 575.0346680, -412.7276306, 697.5901489, -1039.5888672, 987.7622681

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3129426, upper bound: 1379.3087765
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3129426, upper bound: 1379.3087765
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -330.4986267, 991.4775391, -357.2255554, 1078.1440430, -1408.6427002, 1348.7031250
1: -255.2087250, 659.5679932, -276.3755188, 716.4519043, -971.6606445, 935.9434814
2: -276.4482117, 626.2039795, -298.7750854, 681.6100464, -958.0582275, 924.9790649
3: -277.6385803, 803.9555664, -300.8136292, 872.3154297, -1149.9538574, 1104.7691650
4: -402.5508728, 678.9116211, -436.0211792, 738.5900269, -1141.1406250, 1114.9327393

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3127207, upper bound: 1379.3119262
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3127207, upper bound: 1379.3119262
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -330.4986267, 991.4775391, -357.4303284, 1078.8862305, -1409.3848877, 1348.9077148
1: -255.2087250, 659.5679932, -276.5083008, 716.9343872, -972.1430664, 936.0762939
2: -276.4482117, 626.2039795, -298.9260864, 682.0141602, -958.4623413, 925.1300049
3: -277.6385803, 803.9555664, -300.9581909, 872.9194946, -1150.5579834, 1104.9138184
4: -402.5508728, 678.9116211, -436.2965393, 739.0164795, -1141.5672607, 1115.2080078

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3129241, upper bound: 1379.3119521
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3129241, upper bound: 1379.3119521
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -329.3869019, 988.8242798, -357.2255554, 1078.1440430, -1407.5308838, 1346.0498047
1: -254.5123901, 657.7211914, -276.3755188, 716.4519043, -970.9642944, 934.0966797
2: -275.7297668, 624.6491699, -298.7750854, 681.6100464, -957.3396606, 923.4242554
3: -277.0366516, 801.6262817, -300.8136292, 872.3154297, -1149.3520508, 1102.4399414
4: -401.4483948, 677.2485352, -436.0211792, 738.5900269, -1140.0382080, 1113.2696533

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -329.3869019, 988.8242798, -357.4303284, 1078.8862305, -1408.2731934, 1346.2545166
1: -254.5123901, 657.7211914, -276.5083008, 716.9343872, -971.4467773, 934.2294922
2: -275.7297668, 624.6491699, -298.9260864, 682.0141602, -957.7437744, 923.5752563
3: -277.0366516, 801.6262817, -300.9581909, 872.9194946, -1149.9561768, 1102.5844727
4: -401.4483948, 677.2485352, -436.2965393, 739.0164795, -1140.4647217, 1113.5447998

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -306.1796875, 920.7984619, -343.4433899, 1032.1733398, -1338.3530273, 1264.2418213
1: -236.4028320, 611.3881836, -265.3397827, 686.2389526, -922.6417847, 876.7279663
2: -256.1378784, 580.9299316, -287.6015320, 652.1240845, -908.2619629, 868.5314941
3: -257.1143799, 744.7932739, -288.3700867, 835.9073486, -1093.0216064, 1033.1633301
4: -373.3532104, 629.6370239, -418.8767395, 706.8369751, -1080.1901855, 1048.5136719

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3067151, upper bound: 1379.3091635
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3065870, upper bound: 1379.3049031
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -306.9619141, 923.1371460, -338.5450439, 1018.5039673, -1325.4655762, 1261.6820068
1: -237.0007782, 612.9458008, -261.4831238, 676.6633301, -913.6641235, 874.4289551
2: -256.7778015, 582.4044800, -283.4747620, 643.1868286, -899.9645386, 865.8792725
3: -257.7658997, 746.6986694, -284.5165710, 824.3845825, -1082.1505127, 1031.2152100
4: -374.2943420, 631.2342529, -413.0089111, 697.0687256, -1071.3630371, 1044.2431641

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3111583, upper bound: 1379.3107027
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3111583, upper bound: 1379.3107027
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -306.9619141, 923.1371460, -388.1766663, 1174.0740967, -1481.0357666, 1311.3134766
1: -237.0007782, 612.9458008, -299.2076721, 777.8089600, -1014.8097534, 912.1534424
2: -256.7778015, 582.4044800, -324.1359863, 741.1026611, -997.8804321, 906.5404663
3: -257.7658997, 746.6986694, -327.2005615, 945.7882690, -1203.5541992, 1073.8991699
4: -374.2943420, 631.2342529, -472.9415283, 802.3632202, -1176.6575928, 1104.1757812

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.2976147, upper bound: 1379.3012550
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.2975431, upper bound: 1379.3016863
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -306.9619141, 923.1371460, -380.3041077, 1145.3005371, -1452.2619629, 1303.4412842
1: -237.0007782, 612.9458008, -293.4048157, 759.6132202, -996.6140137, 906.3505859
2: -256.7778015, 582.4044800, -317.7694092, 723.8736572, -980.6514282, 900.1737671
3: -257.7658997, 746.6986694, -320.7732239, 924.9089966, -1182.6749268, 1067.4719238
4: -374.2943420, 631.2342529, -462.4606323, 784.6680908, -1158.9622803, 1093.6945801

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.2976147, upper bound: 1379.3024021
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.2975431, upper bound: 1379.3041334
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -309.8525391, 931.0969849, -331.3227234, 997.0710449, -1306.9233398, 1262.4196777
1: -239.4122009, 619.2246704, -256.2861633, 663.1353149, -902.5474854, 875.5108643
2: -260.0131836, 588.1212769, -278.1575928, 630.3093262, -890.3225098, 866.2788696
3: -260.3229980, 754.3041992, -278.5224609, 807.4669189, -1067.7899170, 1032.8265381
4: -378.5709534, 637.1143799, -405.0981750, 683.0009155, -1061.5717773, 1042.2124023

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3113184, upper bound: 1379.3097696
time: 0.78 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3117309, upper bound: 1379.3112565
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -302.1528625, 908.4601440, -324.6323853, 976.5526123, -1278.7054443, 1233.0924072
1: -233.5689850, 604.1680908, -251.1794281, 650.0430908, -883.6120605, 855.3474731
2: -253.8106232, 574.1499023, -273.0667725, 617.6456909, -871.4562988, 847.2165527
3: -253.6796265, 735.7387695, -273.0276794, 791.4637451, -1045.1431885, 1008.7664185
4: -369.5389709, 621.9447021, -397.7192993, 669.3722534, -1038.9112549, 1019.6640015

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107360, upper bound: 1379.3097552
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107817, upper bound: 1379.3112122
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -338.6366882, 1016.7068481, -339.4212341, 1019.9321289, -1358.5686035, 1356.1279297
1: -261.6174927, 676.3482056, -262.2966614, 678.3554077, -939.9729004, 938.6448364
2: -283.6488647, 642.2862549, -284.3731689, 644.3491821, -927.9980469, 926.6594238
3: -284.4909973, 824.0526123, -285.1608276, 826.3839111, -1110.8746338, 1109.2132568
4: -413.0118713, 696.1966553, -414.1601562, 698.4363403, -1111.4482422, 1110.3568115

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3108649, upper bound: 1379.3107882
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107655, upper bound: 1379.3107655
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107655, upper bound: 1379.3107655
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -326.4597168, 980.4669800, -330.7182922, 993.5940552, -1320.0535889, 1311.1851807
1: -252.3598022, 652.2255249, -255.6744690, 661.3659058, -913.7257080, 907.9000244
2: -273.8147583, 619.6168213, -277.7642517, 628.0420532, -901.8568115, 897.3811035
3: -274.0776367, 794.5393677, -277.8717651, 805.6914062, -1079.7690430, 1072.4111328
4: -398.6195068, 671.7106323, -404.5334167, 680.7325439, -1079.3518066, 1076.2440186

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107655, upper bound: 1379.3107655
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107655, upper bound: 1379.3107655
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -309.8525391, 931.0969849, -371.2921448, 1118.4571533, -1428.3095703, 1302.3890381
1: -239.4122009, 619.2246704, -286.7817993, 742.4371338, -981.8493652, 906.0064697
2: -260.0131836, 588.1212769, -310.7084961, 707.2346802, -967.2478638, 898.8297729
3: -260.3229980, 754.3041992, -313.2934875, 903.8030396, -1164.1258545, 1067.5976562
4: -378.5709534, 637.1143799, -452.1213379, 766.5575562, -1145.1285400, 1089.2357178

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3121311, upper bound: 1379.3097641
time: 0.59 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3142621, upper bound: 1379.3112536
time: 0.62 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -302.1528625, 908.4601440, -365.9072571, 1102.1146240, -1404.2674561, 1274.3670654
1: -233.5689850, 604.1680908, -282.5370178, 731.8400879, -965.4090576, 886.7050781
2: -253.8106232, 574.1499023, -306.5764465, 697.0232544, -950.8338623, 880.7262573
3: -253.6796265, 735.7387695, -308.8668213, 890.9780884, -1144.6575928, 1044.6055908
4: -369.5389709, 621.9447021, -446.3370667, 755.5380249, -1125.0767822, 1068.2812500

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3144039, upper bound: 1379.3097596
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3146615, upper bound: 1379.3112393
time: 0.63 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -338.6366882, 1016.7068481, -374.1252136, 1126.3613281, -1464.9979248, 1390.8319092
1: -261.6174927, 676.3482056, -288.9686890, 747.7129517, -1009.3304443, 965.3168945
2: -283.6488647, 642.2862549, -313.0040588, 711.9119263, -995.5607910, 955.2902832
3: -284.4909973, 824.0526123, -315.6856384, 910.7124023, -1195.2028809, 1139.7381592
4: -413.0118713, 696.1966553, -455.5629272, 771.8667603, -1184.8786621, 1151.7595215

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3139313, upper bound: 1379.3107881
time: 0.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3139313, upper bound: 1379.3107881
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -326.4597168, 980.4669800, -364.3133240, 1096.5434570, -1423.0030518, 1344.7802734
1: -252.3598022, 652.2255249, -281.3294373, 728.3119507, -980.6717529, 933.5549316
2: -273.8147583, 619.6168213, -305.3601074, 693.2930908, -967.1078491, 924.9768677
3: -274.0776367, 794.5393677, -307.5005798, 887.0524902, -1161.1301270, 1102.0399170
4: -398.6195068, 671.7106323, -444.5100098, 751.7509155, -1150.3702393, 1116.2205811

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3139313, upper bound: 1379.3107881
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3139313, upper bound: 1379.3107881
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -396.2608337, 1196.7564697, -341.9951172, 1037.6551514, -1433.9158936, 1538.7513428
1: -305.4994812, 793.0464478, -264.7128296, 688.5387573, -994.0381470, 1057.7591553
2: -329.9638977, 755.2761841, -286.7207031, 655.0792236, -985.0430908, 1041.9968262
3: -333.6933594, 965.4001465, -288.1570129, 837.0134277, -1170.7067871, 1253.5570068
4: -481.9981995, 818.2001953, -419.3616333, 708.7404785, -1190.7386475, 1237.5617676

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3153896, upper bound: 1379.3160321
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3153896, upper bound: 1379.3164489
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -417.0616150, 1260.1273193, -344.8499146, 1038.0190430, -1455.0806885, 1604.9771729
1: -321.8710938, 835.3523560, -266.2912903, 689.6809082, -1011.5519409, 1101.6436768
2: -347.5267639, 795.7993774, -288.2132568, 655.4262695, -1002.9529419, 1084.0124512
3: -351.6140137, 1016.5037842, -290.1980286, 840.2213135, -1191.8353271, 1306.7016602
4: -507.3585510, 862.1142578, -420.2829590, 710.1715088, -1217.5300293, 1282.3970947

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3160443, upper bound: 1379.3160449
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3160443, upper bound: 1379.3164611
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -358.9494324, 1080.4765625, -341.9951172, 1037.6551514, -1396.6046143, 1422.4715576
1: -276.8607788, 717.1351318, -264.7128296, 688.5387573, -965.3995361, 981.8479614
2: -299.9561462, 682.3637695, -286.7207031, 655.0792236, -955.0354004, 969.0844727
3: -302.3872375, 873.5123901, -288.1570129, 837.0134277, -1139.4005127, 1161.6694336
4: -437.0062256, 739.5862427, -419.3616333, 708.7404785, -1145.7467041, 1158.9473877

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3153934, upper bound: 1379.3144178
time: 0.83 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3153934, upper bound: 1379.3146681
time: 0.64 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -379.9714966, 1144.7601318, -344.8499146, 1038.0190430, -1417.9902344, 1489.6098633
1: -293.4426575, 759.8486938, -266.2912903, 689.6809082, -983.1235352, 1026.1400146
2: -317.7919312, 723.5714111, -288.2132568, 655.4262695, -973.2182007, 1011.7846069
3: -320.4882202, 925.2454224, -290.1980286, 840.2213135, -1160.7094727, 1215.4432373
4: -462.7405090, 784.2577515, -420.2829590, 710.1715088, -1172.9119873, 1204.5406494

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 44

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3160467, upper bound: 1379.3144304
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3160467, upper bound: 1379.3148174
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -403.8922729, 1220.3937988, -310.7132263, 934.2433472, -1338.1353760, 1531.1070557
1: -311.7599182, 809.2849121, -240.0838165, 621.5558472, -933.3157959, 1049.3687744
2: -336.7437744, 770.9967041, -260.8661499, 590.4680176, -927.2117920, 1031.8627930
3: -340.5332947, 984.2583618, -261.3890076, 757.1285400, -1097.6618652, 1245.6473389
4: -491.4710999, 835.1752319, -379.9787903, 639.9233398, -1131.3944092, 1215.1540527

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3138696, upper bound: 1379.3138696
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3138696, upper bound: 1379.3138696
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -400.5582581, 1209.1313477, -332.9273071, 1000.2274170, -1400.7856445, 1542.0585938
1: -309.2459106, 801.8248291, -257.1116333, 665.1521606, -974.3980713, 1058.9365234
2: -334.0672302, 763.7458496, -278.9152527, 631.8048096, -965.8720093, 1042.6611328
3: -337.7069397, 975.6784058, -280.1039124, 810.5829468, -1148.2899170, 1255.7822266
4: -487.4332581, 827.6034546, -406.2934875, 684.9226074, -1172.3558350, 1233.8969727

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3164335, upper bound: 1379.3150410
time: 0.74 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3164626, upper bound: 1379.3164626
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -365.9101257, 1102.1760254, -310.7132263, 934.2433472, -1300.1531982, 1412.8892822
1: -282.6039734, 731.6886597, -240.0838165, 621.5558472, -904.1597900, 971.7724609
2: -306.2143860, 697.0967407, -260.8661499, 590.4680176, -896.6823730, 957.9627686
3: -308.6095581, 890.5489502, -261.3890076, 757.1285400, -1065.7380371, 1151.9379883
4: -445.5848389, 755.4850464, -379.9787903, 639.9233398, -1085.5081787, 1135.4638672

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3138845, upper bound: 1379.3136917
time: 1.06 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3138845, upper bound: 1379.3147958
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -364.3781433, 1096.8315430, -332.9273071, 1000.2274170, -1364.6051025, 1429.7587891
1: -281.4856873, 728.1896362, -257.1116333, 665.1521606, -946.6378174, 985.3012695
2: -305.0417480, 693.4342041, -278.9152527, 631.8048096, -936.8464966, 972.3493652
3: -307.3561401, 886.7445679, -280.1039124, 810.5829468, -1117.9390869, 1166.8485107
4: -443.8823547, 751.8013306, -406.2934875, 684.9226074, -1128.8049316, 1158.0948486

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3160467, upper bound: 1379.3144302
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3160467, upper bound: 1379.3148168
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -372.4899597, 1121.8439941, -308.7442932, 928.1518555, -1300.6418457, 1430.5882568
1: -287.6989746, 744.7673340, -238.6190033, 617.2242432, -904.9231567, 983.3863525
2: -311.6971741, 709.2943115, -259.1538391, 586.3430786, -898.0402222, 968.4481201
3: -314.2621765, 906.7218628, -259.4469299, 751.8046875, -1066.0668945, 1166.1688232
4: -453.6030579, 768.7456665, -377.3311157, 635.2277222, -1088.8308105, 1146.0766602

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112489, upper bound: 1379.3116291
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112498, upper bound: 1379.3142737
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -376.0675354, 1131.9921875, -337.4628906, 1013.5072632, -1389.5748291, 1469.4550781
1: -290.4291687, 751.5345459, -260.7793884, 674.1880493, -964.6170654, 1012.3139648
2: -314.5764465, 715.4015503, -282.7308044, 640.3267212, -954.9030762, 998.1323242
3: -317.2394104, 915.4342651, -283.5455017, 821.3900757, -1138.6295166, 1198.9797363
4: -457.9020081, 775.5850220, -411.6838379, 694.1754150, -1152.0773926, 1187.2687988

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107959, upper bound: 1379.3116332
time: 0.64 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107967, upper bound: 1379.3142044
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -367.5087891, 1106.7188721, -301.3136292, 906.4955444, -1274.0043945, 1408.0321045
1: -283.7688293, 734.9821167, -232.9978180, 602.8118896, -886.5806885, 967.9799194
2: -307.9126892, 699.8724976, -253.1820831, 572.9878540, -880.9005127, 953.0545654
3: -310.1590576, 894.9008789, -253.1036987, 734.0252075, -1044.1840820, 1148.0045166
4: -448.2899475, 758.5822144, -368.6441345, 620.7084961, -1068.9984131, 1127.2263184

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107878, upper bound: 1379.3130251
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107878, upper bound: 1379.3141847
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -366.4383240, 1102.5051270, -325.4103394, 977.6976318, -1344.1359863, 1427.9155273
1: -282.9191589, 732.3779297, -251.6222229, 650.3461304, -933.2652588, 984.0001221
2: -307.0696716, 697.0083008, -273.0235291, 617.9514771, -925.0210571, 970.0318604
3: -309.1752319, 892.1021118, -273.3029175, 792.2030640, -1101.3781738, 1165.4050293
4: -447.0045471, 755.7218018, -397.4471436, 669.9598999, -1116.9644775, 1153.1688232

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107878, upper bound: 1379.3130252
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107878, upper bound: 1379.3141847
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -372.4899597, 1121.8439941, -355.7578430, 1071.6461182, -1444.1361084, 1477.6018066
1: -287.6989746, 744.7673340, -274.6693420, 711.2793579, -998.9783325, 1019.4366455
2: -311.6971741, 709.2943115, -297.6510620, 677.3325195, -989.0296631, 1006.9453125
3: -314.2621765, 906.7218628, -300.0554810, 866.0223389, -1180.2844238, 1206.7773438
4: -453.6030579, 768.7456665, -433.2348328, 734.0989990, -1187.7020264, 1201.9804688

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3125070, upper bound: 1379.3117859
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3136802, upper bound: 1379.3144432
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -376.0675354, 1131.9921875, -371.5267334, 1117.8204346, -1493.8879395, 1503.5189209
1: -290.4291687, 751.5345459, -286.9714355, 742.1945190, -1032.6235352, 1038.5058594
2: -314.5764465, 715.4015503, -310.8474121, 706.5394897, -1021.1158447, 1026.2490234
3: -317.2394104, 915.4342651, -313.5118713, 904.1057739, -1221.3449707, 1228.9461670
4: -457.9020081, 775.5850220, -452.2882690, 766.1699219, -1224.0717773, 1227.8732910

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3139343, upper bound: 1379.3118236
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3148153, upper bound: 1379.3144458
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -367.5087891, 1106.7188721, -347.3109436, 1046.0330811, -1413.5418701, 1454.0297852
1: -283.7688293, 734.9821167, -268.0981140, 694.2556763, -978.0244751, 1003.0802002
2: -307.9126892, 699.8724976, -290.6002808, 661.3585205, -969.2711792, 990.4727783
3: -310.1590576, 894.9008789, -292.8208923, 845.1118774, -1155.2707520, 1187.7218018
4: -448.2899475, 758.5822144, -422.9148560, 716.7508545, -1165.0407715, 1181.4970703

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3136834, upper bound: 1379.3137113
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3136834, upper bound: 1379.3148228
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -366.4383240, 1102.5051270, -360.0957336, 1083.4771729, -1449.9155273, 1462.6008301
1: -282.9191589, 732.3779297, -278.2261353, 719.3440552, -1002.2631836, 1010.6040649
2: -307.0696716, 697.0083008, -301.5259399, 685.0209351, -992.0905151, 998.5341797
3: -309.1752319, 892.1021118, -303.8679810, 876.0976562, -1185.2725830, 1195.9699707
4: -447.0045471, 755.7218018, -438.5733337, 742.8681030, -1189.8726807, 1194.2951660

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3147368, upper bound: 1379.3137204
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3147368, upper bound: 1379.3148295
time: 0.68 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 6.32 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3115545
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3115545
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3134895, upper bound: 1379.3115570
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3134895, upper bound: 1379.3115571
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3081463, upper bound: 1379.3005450
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3109062, upper bound: 1379.3049054
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3081463, upper bound: 1379.3012371
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3109062, upper bound: 1379.3061660
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3129426, upper bound: 1379.3086822
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3129426, upper bound: 1379.3086822
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3129426, upper bound: 1379.3087765
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3129426, upper bound: 1379.3087765
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3127207, upper bound: 1379.3119262
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3127207, upper bound: 1379.3119262
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3129241, upper bound: 1379.3119521
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3129241, upper bound: 1379.3119521
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3067151, upper bound: 1379.3091635
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3065870, upper bound: 1379.3049031
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3111583, upper bound: 1379.3107027
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3111583, upper bound: 1379.3107027
NS_A1_B2_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.2976147, upper bound: 1379.3012550
NS_A1_B2_A1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.2975431, upper bound: 1379.3016863
NS_A1_B2_A1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.2976147, upper bound: 1379.3024021
NS_A1_B2_A1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.2975431, upper bound: 1379.3041334
NS_A1_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3113184, upper bound: 1379.3097696
NS_A1_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3117309, upper bound: 1379.3112565
NS_A1_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107360, upper bound: 1379.3097552
NS_A1_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107817, upper bound: 1379.3112122
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107655, upper bound: 1379.3107655
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107655, upper bound: 1379.3107655
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107655, upper bound: 1379.3107655
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107655, upper bound: 1379.3107655
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3121311, upper bound: 1379.3097641
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3142621, upper bound: 1379.3112536
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3144039, upper bound: 1379.3097596
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3146615, upper bound: 1379.3112393
NS_A1_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3139313, upper bound: 1379.3107881
NS_A1_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3139313, upper bound: 1379.3107881
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3139313, upper bound: 1379.3107881
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3139313, upper bound: 1379.3107881
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3153896, upper bound: 1379.3160321
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3153896, upper bound: 1379.3164489
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3160443, upper bound: 1379.3160449
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3160443, upper bound: 1379.3164611
NS_A2_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3153934, upper bound: 1379.3144178
NS_A2_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3153934, upper bound: 1379.3146681
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3160467, upper bound: 1379.3144304
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3160467, upper bound: 1379.3148174
NS_A2_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3138696, upper bound: 1379.3138696
NS_A2_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3138696, upper bound: 1379.3138696
NS_A2_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3164335, upper bound: 1379.3150410
NS_A2_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3164626, upper bound: 1379.3164626
NS_A2_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3138845, upper bound: 1379.3136917
NS_A2_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3138845, upper bound: 1379.3147958
NS_A2_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3160467, upper bound: 1379.3144302
NS_A2_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3160467, upper bound: 1379.3148168
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3112489, upper bound: 1379.3116291
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3112498, upper bound: 1379.3142737
NS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107959, upper bound: 1379.3116332
NS_A2_B2_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107967, upper bound: 1379.3142044
NS_A2_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107878, upper bound: 1379.3130251
NS_A2_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107878, upper bound: 1379.3141847
NS_A2_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107878, upper bound: 1379.3130252
NS_A2_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3107878, upper bound: 1379.3141847
NS_A2_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3125070, upper bound: 1379.3117859
NS_A2_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3136802, upper bound: 1379.3144432
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3139343, upper bound: 1379.3118236
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3148153, upper bound: 1379.3144458
NS_A2_B2_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3136834, upper bound: 1379.3137113
NS_A2_B2_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3136834, upper bound: 1379.3148228
NS_A2_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3147368, upper bound: 1379.3137204
NS_A2_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 6.32
Output dim: 0, lower bound: -1379.3147368, upper bound: 1379.3148295

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -290.9761658, 872.3879395, -343.4040222, 1038.0156250, -1328.9914551, 1215.7917480
1: -224.7227936, 580.5257568, -266.0473328, 689.7423096, -914.4650879, 846.5730591
2: -244.0269318, 551.0678711, -287.8786011, 656.5622559, -900.5891724, 838.9464722
3: -244.4068604, 707.5724487, -289.3753357, 839.5045166, -1083.9111328, 996.9476929
4: -355.0779114, 597.2135620, -419.9307861, 711.3106079, -1066.3885498, 1017.1441650

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3113983
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3115546
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -291.4514465, 873.8728638, -343.4040222, 1038.0156250, -1329.4667969, 1217.2768555
1: -225.1119995, 581.5972900, -266.0473328, 689.7423096, -914.8543091, 847.6446533
2: -244.4626923, 552.0025635, -287.8786011, 656.5622559, -901.0249634, 839.8811646
3: -244.7747650, 708.8605347, -289.3753357, 839.5045166, -1084.2791748, 998.2358398
4: -355.7448425, 598.2257690, -419.9307861, 711.3106079, -1067.0554199, 1018.1564331

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3113983
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3115545
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -290.9761658, 872.3879395, -343.5857239, 1038.6292725, -1329.6054688, 1215.9732666
1: -224.7227936, 580.5257568, -266.1556396, 690.1550293, -914.8778076, 846.6813965
2: -244.0269318, 551.0678711, -287.9777527, 656.9071045, -900.9340210, 839.0456543
3: -244.4068604, 707.5724487, -289.4987488, 840.0092773, -1084.4160156, 997.0711670
4: -355.0779114, 597.2135620, -420.1456299, 711.6771240, -1066.7550049, 1017.3590698

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3112437
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3115571
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -291.4514465, 873.8728638, -343.5857239, 1038.6292725, -1330.0806885, 1217.4584961
1: -225.1119995, 581.5972900, -266.1556396, 690.1550293, -915.2669678, 847.7529297
2: -244.4626923, 552.0025635, -287.9777527, 656.9071045, -901.3698120, 839.9803467
3: -244.7747650, 708.8605347, -289.4987488, 840.0092773, -1084.7840576, 998.3592529
4: -355.7448425, 598.2257690, -420.1456299, 711.6771240, -1067.4218750, 1018.3713379

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3112437
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3130950, upper bound: 1379.3115545
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -262.1818237, 787.8851929, -309.7383118, 937.9046631, -1200.0864258, 1097.6234131
1: -202.6821136, 523.8880615, -239.9981537, 623.5272827, -826.2093506, 763.8861694
2: -220.9915771, 497.8405151, -260.6061707, 593.3348389, -814.3264160, 758.4465942
3: -220.6054688, 637.9967041, -261.0645142, 758.5128784, -979.1183472, 899.0612183
4: -321.3799744, 539.3452148, -380.5280151, 642.6242065, -964.0040894, 919.8732300

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3049866, upper bound: 1379.2984296
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3049866, upper bound: 1379.3005450
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -257.1814880, 773.1014404, -301.8698425, 912.5613403, -1169.7426758, 1074.9711914
1: -198.7880554, 514.1872559, -233.6263885, 606.6702271, -805.4582520, 747.8136597
2: -216.5248108, 487.9078674, -253.8757019, 576.9054565, -793.4302979, 741.7834473
3: -215.8149414, 626.3768921, -254.0756683, 738.2924194, -954.1073608, 880.4525757
4: -315.4738770, 528.4608154, -370.5512390, 624.8896484, -940.3635254, 899.0120850

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3050057, upper bound: 1379.3027789
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3050057, upper bound: 1379.3049054
time: 0.58 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -262.1818237, 787.8851929, -331.0841370, 1000.6527710, -1262.8344727, 1118.9693604
1: -202.6821136, 523.8880615, -256.5328979, 665.5446777, -868.2267456, 780.4209595
2: -220.9915771, 497.8405151, -278.2730713, 633.2014771, -854.1930542, 776.1135864
3: -220.6054688, 637.9967041, -279.1019592, 809.8139038, -1030.4194336, 917.0986328
4: -321.3799744, 539.3452148, -406.0774536, 686.2113037, -1007.5911255, 945.4226685

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3076128, upper bound: 1379.3003933
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3081463, upper bound: 1379.3012357
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -257.1814880, 773.1014404, -323.4226990, 976.5148926, -1233.6960449, 1096.5241699
1: -198.7880554, 514.1872559, -250.3128204, 649.3331299, -848.1212158, 764.5000610
2: -216.5248108, 487.9078674, -271.6145325, 617.3827515, -833.9075928, 759.5223999
3: -215.8149414, 626.3768921, -272.3532410, 790.4207764, -1006.2355957, 898.7301025
4: -315.4738770, 528.4608154, -396.4238892, 669.0435791, -984.5174561, 924.8847046

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3116200, upper bound: 1379.3060950
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3115239, upper bound: 1379.3061014
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -277.3322754, 831.5155640, -337.2131958, 1019.0330811, -1296.3653564, 1168.7287598
1: -214.2261353, 553.3424683, -261.2059937, 678.0353394, -892.2614746, 814.5484009
2: -232.8986664, 525.2744141, -283.1549988, 644.7484131, -877.6470947, 808.4294434
3: -232.7295380, 674.3340454, -283.8608704, 825.1578979, -1057.8869629, 958.1949463
4: -338.8400879, 569.2324829, -413.5165100, 698.6024170, -1037.4423828, 982.7489014

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -277.2113342, 832.0214844, -337.2131958, 1019.0330811, -1296.2443848, 1169.2346191
1: -214.2859802, 553.5287476, -261.2059937, 678.0353394, -892.3212891, 814.7347412
2: -232.9678955, 525.6881104, -283.1549988, 644.7484131, -877.7163086, 808.8431396
3: -232.9547882, 674.4040527, -283.8608704, 825.1578979, -1058.1124268, 958.2648926
4: -338.8841858, 569.6757812, -413.5165100, 698.6024170, -1037.4865723, 983.1921997

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -277.3322754, 831.5155640, -336.4512329, 1017.4390869, -1294.7712402, 1167.9666748
1: -214.2261353, 553.3424683, -260.7100830, 676.7816162, -891.0077515, 814.0524292
2: -232.8986664, 525.2744141, -282.6421204, 643.8250122, -876.7236938, 807.9165039
3: -232.7295380, 674.3340454, -283.4180908, 823.4862671, -1056.2154541, 957.7520752
4: -338.8400879, 569.2324829, -412.7276306, 697.5901489, -1036.4300537, 981.9600830

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -277.2113342, 832.0214844, -336.4512329, 1017.4390869, -1294.6502686, 1168.4725342
1: -214.2859802, 553.5287476, -260.7100830, 676.7816162, -891.0675049, 814.2388306
2: -232.9678955, 525.6881104, -282.6421204, 643.8250122, -876.7929077, 808.3302002
3: -232.9547882, 674.4040527, -283.4180908, 823.4862671, -1056.4407959, 957.8221436
4: -338.8841858, 569.6757812, -412.7276306, 697.5901489, -1036.4742432, 982.4033813

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -327.8687744, 983.1420288, -357.2255554, 1078.1440430, -1406.0125732, 1340.3675537
1: -253.1431732, 654.0692139, -276.3755188, 716.4519043, -969.5949097, 930.4447021
2: -274.2592773, 620.9936523, -298.7750854, 681.6100464, -955.8692017, 919.7687378
3: -275.4108276, 797.2367554, -300.8136292, 872.3154297, -1147.7260742, 1098.0504150
4: -399.2521362, 673.2586060, -436.0211792, 738.5900269, -1137.8420410, 1109.2796631

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -328.2428284, 984.4442749, -357.2255554, 1078.1440430, -1406.3868408, 1341.6697998
1: -253.4487305, 654.9545288, -276.3755188, 716.4519043, -969.9006348, 931.3300781
2: -274.5809631, 621.7883301, -298.7750854, 681.6100464, -956.1909180, 920.5634155
3: -275.7025757, 798.3148193, -300.8136292, 872.3154297, -1148.0177002, 1099.1284180
4: -399.7750549, 674.1204224, -436.0211792, 738.5900269, -1138.3649902, 1110.1416016

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -327.8687744, 983.1420288, -357.4303284, 1078.8862305, -1406.7548828, 1340.5722656
1: -253.1431732, 654.0692139, -276.5083008, 716.9343872, -970.0773926, 930.5775146
2: -274.2592773, 620.9936523, -298.9260864, 682.0141602, -956.2733154, 919.9197388
3: -275.4108276, 797.2367554, -300.9581909, 872.9194946, -1148.3303223, 1098.1949463
4: -399.2521362, 673.2586060, -436.2965393, 739.0164795, -1138.2685547, 1109.5548096

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 34

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 44

### Candidate
type: B, layer: 1, pos: 47

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -328.2428284, 984.4442749, -357.4303284, 1078.8862305, -1407.1290283, 1341.8743896
1: -253.4487305, 654.9545288, -276.5083008, 716.9343872, -970.3831177, 931.4628296
2: -274.5809631, 621.7883301, -298.9260864, 682.0141602, -956.5950317, 920.7144165
3: -275.7025757, 798.3148193, -300.9581909, 872.9194946, -1148.6219482, 1099.2729492
4: -399.7750549, 674.1204224, -436.2965393, 739.0164795, -1138.7915039, 1110.4168701

Time for backsubstitution: 1.76 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.55 + 417.99 = 421.54 seconds
