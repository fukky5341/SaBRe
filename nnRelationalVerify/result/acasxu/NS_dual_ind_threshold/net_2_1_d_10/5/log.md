## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_1.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 5)
Time budget: 420 seconds
Split limit: 100
Threshold: 1442.1242243135432


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-710.2344360, 905.8535156, -710.2344360, 905.8535156, -1616.0878906, 1616.0878906)
1: (-562.7462158, 727.9412842, -562.7462158, 727.9412842, -1290.6872559, 1290.6872559)
2: (-488.6404724, 716.1848145, -488.6404724, 716.1848145, -1204.8249512, 1204.8249512)
3: (-664.5035400, 884.3118286, -664.5035400, 884.3118286, -1548.8153076, 1548.8153076)
4: (-654.3821411, 970.1540527, -654.3821411, 970.1540527, -1624.5361328, 1624.5361328)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 2.27 = 3.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1442.1386457, upper bound: 1442.1386457

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1377453, upper bound: 1442.1375843
time: 0.85 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1377091, upper bound: 1442.1377091
time: 0.95 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.92 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 0, lower bound: -1442.1377453, upper bound: 1442.1375843
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.92
Output dim: 0, lower bound: -1442.1377091, upper bound: 1442.1377091

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -658.6774292, 839.6923218, -684.0079346, 872.0634766, -1530.7408447, 1523.7001953
1: -521.0994263, 674.0761719, -541.5390015, 700.4181519, -1221.5174561, 1215.6149902
2: -452.7773132, 663.3363037, -470.3756714, 689.1622925, -1141.9394531, 1133.7119141
3: -615.9075317, 818.6604004, -639.7347412, 850.7722778, -1466.6796875, 1458.3950195
4: -606.0281982, 898.3854980, -629.7462158, 933.4438477, -1539.4720459, 1528.1317139

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1363261, upper bound: 1442.1365320
time: 0.94 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1377267, upper bound: 1442.1375642
time: 0.81 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -708.1145020, 901.7516479, -678.8015747, 865.9365234, -1574.0510254, 1580.5532227
1: -559.8621826, 723.6748047, -537.6331787, 695.3912354, -1255.2534180, 1261.3079834
2: -486.3637085, 712.5987549, -466.7797241, 684.6027832, -1170.9665527, 1179.3781738
3: -661.5999146, 878.5103149, -635.0512695, 844.5623169, -1506.1622314, 1513.5615234
4: -650.9104614, 965.5377197, -625.0939941, 927.4983521, -1578.4085693, 1590.6317139

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1363360, upper bound: 1442.1366266
time: 1.15 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1363360, upper bound: 1442.1376885
time: 1.07 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.57 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -1442.1363261, upper bound: 1442.1365320
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -1442.1377267, upper bound: 1442.1375642
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -1442.1363360, upper bound: 1442.1366266
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.57
Output dim: 0, lower bound: -1442.1363360, upper bound: 1442.1376885

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -634.2990723, 807.0433960, -632.4151611, 806.0200806, -1440.3187256, 1439.4584961
1: -501.4773865, 648.0350952, -500.9259949, 647.3226929, -1148.7999268, 1148.9610596
2: -435.6959229, 637.3959351, -435.0029907, 637.2938232, -1072.9897461, 1072.3988037
3: -592.1683350, 787.1210327, -591.2510986, 786.1735229, -1378.3417969, 1378.3719482
4: -583.0854492, 863.0588379, -582.2637939, 863.4503784, -1446.5358887, 1445.3225098

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362987
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1365320
time: 1.21 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -647.8525391, 826.1896973, -667.0830078, 850.9970093, -1498.8494873, 1493.2727051
1: -512.5456543, 663.2126465, -528.1856689, 683.4586182, -1196.0042725, 1191.3983154
2: -445.3307495, 652.6691284, -458.7611084, 672.5101929, -1117.8409424, 1111.4301758
3: -605.8899536, 805.5497437, -624.0870972, 830.3073120, -1436.1970215, 1429.6368408
4: -596.1401367, 883.9110718, -614.3182373, 910.8562012, -1506.9963379, 1498.2290039

Time for backsubstitution: 1.25 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376250, upper bound: 1442.1374756
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1377125, upper bound: 1442.1375531
time: 0.90 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -684.6113892, 870.2652588, -632.7084961, 806.8378906, -1491.4492188, 1502.9737549
1: -540.8471680, 698.4893188, -501.3168640, 647.8624268, -1188.7093506, 1199.8061523
2: -469.8298035, 687.4624634, -435.1432800, 638.1420898, -1107.9719238, 1122.6057129
3: -638.6475830, 847.9771729, -591.7400513, 786.8341064, -1425.4816895, 1439.7172852
4: -628.7068481, 931.3129272, -582.6887817, 864.8480225, -1493.5544434, 1514.0017090

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361781, upper bound: 1442.1364801
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361900, upper bound: 1442.1364947
time: 0.96 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -696.9959106, 887.9411011, -662.5466309, 845.6762085, -1542.6721191, 1550.4876709
1: -551.1806030, 712.6128540, -524.9022827, 679.1513062, -1230.3319092, 1237.5148926
2: -478.8065796, 701.7162476, -455.7169495, 668.6448975, -1147.4514160, 1157.4332275
3: -651.4031372, 865.1623535, -620.0836182, 824.9993896, -1476.4024658, 1485.2459717
4: -640.8739624, 950.7662964, -610.3912964, 905.8646851, -1546.7385254, 1561.1572266

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376884, upper bound: 1442.1376782
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376782, upper bound: 1442.1376782
time: 1.06 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.67 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362987
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1365320
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 0, lower bound: -1442.1376250, upper bound: 1442.1374756
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 0, lower bound: -1442.1377125, upper bound: 1442.1375531
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 0, lower bound: -1442.1361781, upper bound: 1442.1364801
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 0, lower bound: -1442.1361900, upper bound: 1442.1364947
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 0, lower bound: -1442.1376884, upper bound: 1442.1376782
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.67
Output dim: 0, lower bound: -1442.1376782, upper bound: 1442.1376782

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -600.6600342, 763.0824585, -615.7922974, 784.1719360, -1384.8320312, 1378.8747559
1: -474.4675598, 612.8108521, -487.5634766, 629.8317261, -1104.2989502, 1100.3742676
2: -412.3080444, 602.4686279, -423.4329224, 619.9338379, -1032.2419434, 1025.9014893
3: -560.0338745, 744.3297119, -575.3187866, 764.9167480, -1324.9506836, 1319.6481934
4: -551.6969604, 815.6646118, -566.7147827, 839.8823242, -1391.5793457, 1382.3791504

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362987
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362987
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -819.3673096, 1040.8973389, -616.3311768, 785.0463867, -1604.4135742, 1657.2285156
1: -647.8907471, 837.8936768, -488.0214844, 630.5734253, -1278.4641113, 1325.9151611
2: -563.7084351, 823.5830078, -423.7937012, 620.6914673, -1184.3999023, 1247.3767090
3: -765.0805664, 1016.6986694, -575.9228516, 765.7122192, -1530.7927246, 1592.6210938
4: -753.8678589, 1115.0805664, -567.2279053, 840.8419800, -1594.7098389, 1682.3083496

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304347, upper bound: 1442.1332732
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361771, upper bound: 1442.1364044
time: 0.90 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -609.9169922, 776.2276611, -641.3999023, 817.2352905, -1427.1523438, 1417.6273193
1: -482.0694885, 623.1370850, -507.5169373, 656.3531494, -1138.4226074, 1130.6540527
2: -418.8705444, 612.9511108, -440.8409424, 645.6251831, -1064.4953613, 1053.7919922
3: -569.5458374, 756.8814087, -599.5329590, 797.3892212, -1366.9350586, 1356.4143066
4: -560.5777588, 830.0007935, -590.2639160, 874.3798828, -1434.9576416, 1420.2646484

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376146, upper bound: 1442.1374756
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376146, upper bound: 1442.1374716
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -815.8777466, 1040.5150146, -645.8715820, 823.7957153, -1639.6733398, 1686.3865967
1: -646.1906738, 838.4653931, -511.2172546, 661.5861816, -1307.7767334, 1349.6826172
2: -560.7781372, 821.9796753, -444.0385742, 650.9427490, -1211.7209473, 1266.0181885
3: -762.6207275, 1020.5311279, -604.0867920, 803.7704468, -1566.3911133, 1624.6179199
4: -752.3273926, 1111.8787842, -594.6463013, 881.6456299, -1633.9730225, 1706.5247803

Time for backsubstitution: 1.26 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375246
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375163
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -638.5066528, 812.6287231, -614.6423950, 784.0762329, -1422.5826416, 1427.2711182
1: -504.2186279, 651.6078491, -486.9618835, 629.3712158, -1133.5898438, 1138.5697021
2: -438.2008972, 641.9236450, -422.7304688, 620.1271362, -1058.3280029, 1064.6540527
3: -596.1769409, 790.9017334, -575.0344849, 764.3085327, -1360.4854736, 1365.9362793
4: -586.1634521, 869.8279419, -565.9935303, 840.4907227, -1426.6541748, 1435.8214111

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361781, upper bound: 1442.1364801
time: 1.09 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361781, upper bound: 1442.1364801
time: 0.86 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -755.5415649, 959.9692383, -623.4407959, 795.0253906, -1550.5668945, 1583.4099121
1: -594.7479248, 768.2562256, -494.1260681, 638.3646851, -1233.1125488, 1262.3822021
2: -517.1665649, 756.7172852, -428.8882141, 628.8508911, -1146.0173340, 1185.6054688
3: -703.4251709, 931.9551392, -583.1963501, 775.3948364, -1478.8199463, 1515.1514893
4: -691.3053589, 1025.0651855, -574.2864990, 852.3442383, -1543.6496582, 1599.3516846

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361900, upper bound: 1442.1364947
time: 0.98 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361900, upper bound: 1442.1364947
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -670.9417114, 853.5722046, -647.3643799, 825.6750488, -1496.6163330, 1500.9361572
1: -530.1311035, 685.0753784, -512.6741333, 663.1213989, -1193.2524414, 1197.7492676
2: -460.6053162, 674.4154663, -445.1246643, 652.7576294, -1113.3629150, 1119.5401611
3: -626.3978271, 831.6779785, -605.5073242, 805.5136719, -1431.9114990, 1437.1853027
4: -616.4050903, 913.7312622, -596.1513672, 884.2999268, -1500.7049561, 1509.8825684

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375372, upper bound: 1442.1376518
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376782, upper bound: 1442.1376782
time: 0.89 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376782, upper bound: 1442.1376782
time: 1.04 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -773.6660156, 986.9965210, -646.8746338, 825.6110840, -1599.2766113, 1633.8709717
1: -611.3004150, 792.9371948, -512.4340820, 663.1015015, -1274.4018555, 1305.3712158
2: -532.1506348, 782.0020752, -444.8489685, 652.7915649, -1184.9418945, 1226.8508301
3: -724.4769287, 960.9243774, -605.3096313, 805.4251709, -1529.9020996, 1566.2340088
4: -711.2857666, 1059.2414551, -595.8621216, 884.3156738, -1595.6011963, 1655.1035156

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1374097, upper bound: 1442.1374011
time: 1.11 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1372283, upper bound: 1442.1372283
time: 1.07 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.57 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362987
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362987
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1304347, upper bound: 1442.1332732
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1361771, upper bound: 1442.1364044
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1376146, upper bound: 1442.1374756
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1376146, upper bound: 1442.1374716
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375246
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375163
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1361781, upper bound: 1442.1364801
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1361781, upper bound: 1442.1364801
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1361900, upper bound: 1442.1364947
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1361900, upper bound: 1442.1364947
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1376782, upper bound: 1442.1376782
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1376782, upper bound: 1442.1376782
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1374097, upper bound: 1442.1374011
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.57
Output dim: 0, lower bound: -1442.1372283, upper bound: 1442.1372283

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -600.6600342, 763.0824585, -584.8507690, 745.0339966, -1345.6940918, 1347.9332275
1: -474.4675598, 612.8108521, -462.7318726, 597.9927979, -1072.4599609, 1075.5427246
2: -412.3080444, 602.4686279, -402.0532532, 588.7947998, -1001.1028442, 1004.5217285
3: -560.0338745, 744.3297119, -546.4636841, 726.1284180, -1286.1623535, 1290.7932129
4: -551.6969604, 815.6646118, -537.9462891, 797.6225586, -1349.3195801, 1353.6108398

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362924
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362987
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -600.6600342, 763.0824585, -652.5117798, 829.6059570, -1430.2659912, 1415.5942383
1: -474.4675598, 612.8108521, -515.7770386, 665.7797241, -1140.2470703, 1128.5878906
2: -412.3080444, 602.4686279, -447.9968872, 655.7226562, -1068.0307617, 1050.4654541
3: -560.0338745, 744.3297119, -608.9086304, 808.2564697, -1368.2902832, 1353.2380371
4: -551.6969604, 815.6646118, -599.4792480, 888.7031250, -1440.4000244, 1415.1436768

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362924
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362987
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -813.7101440, 1033.9649658, -569.4039307, 726.3779907, -1540.0880127, 1603.3686523
1: -643.4885864, 832.3011475, -450.9070435, 583.0986328, -1226.5871582, 1283.2081299
2: -559.8897095, 818.1563721, -391.7252808, 574.5664062, -1134.4560547, 1209.8815918
3: -759.9449463, 1009.8920898, -532.7806396, 707.8715210, -1467.8164062, 1542.6727295
4: -748.7659302, 1107.7607422, -524.1286621, 778.4970093, -1527.2625732, 1631.8891602

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1330917
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1332732
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -721.2637329, 918.7243042, -654.9114380, 832.8476562, -1554.1113281, 1573.6357422
1: -570.7035522, 739.3852539, -516.6654053, 667.6907959, -1238.3942871, 1256.0506592
2: -496.7719727, 727.8592529, -449.1707764, 657.2110596, -1153.9830322, 1177.0299072
3: -675.4253540, 897.0237427, -610.5296631, 809.9638672, -1485.3889160, 1507.5533447
4: -664.1895752, 985.7763062, -600.5786743, 890.2842407, -1554.4738770, 1586.3547363

Time for backsubstitution: 1.28 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1360526
time: 0.89 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1364044
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -595.4061279, 757.1895142, -611.2731934, 777.6597900, -1373.0657959, 1368.4626465
1: -470.3862915, 607.8668213, -483.2408752, 624.5850220, -1094.9713135, 1091.1076660
2: -408.7581482, 597.8070679, -419.8180237, 614.1383057, -1022.8964844, 1017.6250610
3: -555.6192627, 738.3082275, -570.6395264, 758.7363281, -1314.3554688, 1308.9477539
4: -546.9864502, 809.4470825, -561.9980469, 831.6524658, -1378.6389160, 1371.4450684

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376146, upper bound: 1442.1374716
time: 1.59 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376146, upper bound: 1442.1374716
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -592.6213989, 753.6799316, -810.6115112, 1030.7321777, -1623.3532715, 1564.2912598
1: -468.2486267, 605.1749268, -641.1342163, 829.5994873, -1297.8481445, 1246.3090820
2: -406.8459473, 595.1249390, -557.8076172, 815.5884399, -1222.4343262, 1152.9326172
3: -553.0867920, 734.9586792, -757.4038086, 1006.6968994, -1559.7833252, 1492.3624268
4: -544.4670410, 805.7422485, -746.1348267, 1104.3889160, -1648.8557129, 1551.8770752

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1372466, upper bound: 1442.1370530
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1369479, upper bound: 1442.1365770
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -798.7384033, 1017.9994507, -615.3454590, 783.6099243, -1582.3483887, 1633.3449707
1: -632.4239502, 820.4445801, -486.5945435, 629.3319702, -1261.7556152, 1307.0390625
2: -548.8539429, 804.0437622, -422.7252197, 618.9642944, -1167.8179932, 1226.7690430
3: -746.1676025, 998.6340332, -574.7576904, 764.5104980, -1510.6781006, 1573.3917236
4: -736.3069458, 1087.5543213, -565.9726562, 838.2301636, -1574.5368652, 1653.5267334

Time for backsubstitution: 1.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375163
time: 0.84 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375163
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -800.4096069, 1020.8374634, -818.9315796, 1041.9296875, -1842.3391113, 1839.7687988
1: -633.9373779, 822.7047729, -647.8930664, 838.6157837, -1472.5532227, 1470.5977783
2: -550.1057739, 806.3836060, -563.6307983, 824.5121460, -1374.6176758, 1370.0144043
3: -748.1468506, 1001.3816528, -765.5197754, 1017.6923218, -1765.8391113, 1766.8177490
4: -738.0808716, 1090.7097168, -754.0483398, 1116.5321045, -1854.6126709, 1844.7580566

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 5

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1372768, upper bound: 1442.1371233
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375163
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375163
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -638.5066528, 812.6287231, -585.4176025, 747.0148926, -1385.5212402, 1398.0463867
1: -504.2186279, 651.6078491, -463.3950195, 599.3410034, -1103.5595703, 1115.0029297
2: -438.2008972, 641.9236450, -402.6322021, 590.5078735, -1028.7087402, 1044.5559082
3: -596.1769409, 790.9017334, -547.7038574, 727.7125244, -1323.8894043, 1338.6055908
4: -586.1634521, 869.8279419, -538.7295532, 800.0543213, -1386.2177734, 1408.5570068

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1360213, upper bound: 1442.1364332
time: 0.86 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1359991, upper bound: 1442.1364013
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -638.5066528, 812.6287231, -647.8898926, 824.6820068, -1463.1884766, 1460.5185547
1: -504.2186279, 651.6078491, -512.2518921, 661.5255737, -1165.7440186, 1163.8597412
2: -438.2008972, 641.9236450, -444.9556885, 651.8737793, -1090.0747070, 1086.8793945
3: -596.1769409, 790.9017334, -605.1475830, 803.0385132, -1399.2154541, 1396.0489502
4: -586.1634521, 869.8279419, -595.3835449, 883.6004639, -1469.7639160, 1465.2109375

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1360213, upper bound: 1442.1364332
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1359991, upper bound: 1442.1364013
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -755.5415649, 959.9692383, -585.1091919, 745.6765747, -1501.2180176, 1545.0778809
1: -594.7479248, 768.2562256, -463.1763916, 598.3707886, -1193.1186523, 1231.4324951
2: -517.1665649, 756.7172852, -402.3885803, 589.3985596, -1106.5651855, 1159.1058350
3: -703.4251709, 931.9551392, -547.1062012, 726.5984497, -1430.0235596, 1479.0610352
4: -691.3053589, 1025.0651855, -538.3583984, 798.6243286, -1489.9295654, 1563.4235840

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1359934, upper bound: 1442.1363950
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1359724, upper bound: 1442.1363660
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -755.5415649, 959.9692383, -655.5599365, 834.0987549, -1589.6403809, 1615.5289307
1: -594.7479248, 768.2562256, -518.5196533, 669.2885742, -1264.0362549, 1286.7758789
2: -517.1665649, 756.7172852, -450.3214722, 659.3486328, -1176.5151367, 1207.0388184
3: -703.4251709, 931.9551392, -612.1812744, 812.6201172, -1516.0450439, 1544.1362305
4: -691.3053589, 1025.0651855, -602.6189575, 893.7207642, -1585.0260010, 1627.6840820

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1359934, upper bound: 1442.1364063
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1359724, upper bound: 1442.1363766
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -670.9417114, 853.5722046, -630.9260864, 804.1094971, -1475.0511475, 1484.4981689
1: -530.1311035, 685.0753784, -499.4428101, 645.8069458, -1175.9378662, 1184.5179443
2: -460.6053162, 674.4154663, -433.6595764, 635.6187134, -1096.2239990, 1108.0749512
3: -626.3978271, 831.6779785, -589.7812500, 784.4582520, -1410.8560791, 1421.4592285
4: -616.4050903, 913.7312622, -580.7382812, 861.0503540, -1477.4550781, 1494.4694824

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1373255, upper bound: 1442.1369214
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1373842, upper bound: 1442.1373637
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -670.9417114, 853.5722046, -779.3132935, 993.9079590, -1664.8496094, 1632.8853760
1: -530.1311035, 685.0753784, -616.3373413, 798.9232788, -1329.0544434, 1301.4124756
2: -460.6053162, 674.4154663, -536.3646851, 787.0200195, -1247.6253662, 1210.7800293
3: -626.3978271, 831.6779785, -729.5640259, 968.7051392, -1595.1030273, 1561.2419434
4: -616.4050903, 913.7312622, -717.1352539, 1065.7478027, -1682.1525879, 1630.8663330

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1373255, upper bound: 1442.1369214
time: 0.98 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1373842, upper bound: 1442.1373637
time: 0.81 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -763.0682983, 973.7673950, -598.0422974, 764.6043091, -1527.6726074, 1571.8096924
1: -602.8951416, 782.2650757, -473.8254089, 613.6335449, -1216.5284424, 1256.0904541
2: -524.8816528, 771.5858154, -411.4844666, 604.7447510, -1129.6263428, 1183.0703125
3: -714.6971436, 947.9360962, -560.4733887, 745.1658936, -1459.8629150, 1508.4094238
4: -701.5542603, 1045.1313477, -551.0319214, 819.4207153, -1520.9749756, 1596.1629639

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1372566, upper bound: 1442.1372047
time: 1.17 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1373482, upper bound: 1442.1373869
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -711.4686279, 911.1793213, -681.5071411, 868.4497681, -1579.9184570, 1592.6865234
1: -563.0989380, 731.7827759, -537.8942871, 695.8247681, -1258.9237061, 1269.6770020
2: -490.2361755, 722.9443359, -467.4030151, 685.3465576, -1175.5827637, 1190.3469238
3: -668.6179199, 886.8451538, -636.0354004, 844.2684326, -1512.8861084, 1522.8806152
4: -655.3364258, 979.8602905, -625.3643799, 928.4722290, -1583.8084717, 1605.2246094

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1370713, upper bound: 1442.1369685
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1371828, upper bound: 1442.1371828
time: 0.88 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.22 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362924
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362987
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362924
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1361832, upper bound: 1442.1362987
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1330917
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1332732
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1360526
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1364044
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1376146, upper bound: 1442.1374716
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1376146, upper bound: 1442.1374716
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1372466, upper bound: 1442.1370530
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1369479, upper bound: 1442.1365770
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375163
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375163
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375163
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1376266, upper bound: 1442.1375163
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1360213, upper bound: 1442.1364332
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1359991, upper bound: 1442.1364013
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1360213, upper bound: 1442.1364332
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1359991, upper bound: 1442.1364013
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1359934, upper bound: 1442.1363950
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1359724, upper bound: 1442.1363660
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1359934, upper bound: 1442.1364063
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1359724, upper bound: 1442.1363766
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1373255, upper bound: 1442.1369214
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1373842, upper bound: 1442.1373637
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1373255, upper bound: 1442.1369214
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1373842, upper bound: 1442.1373637
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1372566, upper bound: 1442.1372047
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1373482, upper bound: 1442.1373869
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1370713, upper bound: 1442.1369685
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 3.22
Output dim: 0, lower bound: -1442.1371828, upper bound: 1442.1371828

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -569.1553955, 724.4523926, -584.8507690, 745.0339966, -1314.1894531, 1309.3032227
1: -450.1062012, 581.4873657, -462.7318726, 597.9927979, -1048.0987549, 1044.2192383
2: -391.1219788, 572.4326172, -402.0532532, 588.7947998, -979.9166870, 974.4858398
3: -531.4224854, 706.0717163, -546.4636841, 726.1284180, -1257.5509033, 1252.5354004
4: -523.2495117, 775.3944092, -537.9462891, 797.6225586, -1320.8719482, 1313.3406982

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.09 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1369226, upper bound: 1442.1368913
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1369226, upper bound: 1442.1368913
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -610.8034058, 778.0040283, -584.8507690, 745.0339966, -1355.8369141, 1362.8547363
1: -482.7779541, 624.5186768, -462.7318726, 597.9927979, -1080.7707520, 1087.2503662
2: -419.5393982, 614.3937378, -402.0532532, 588.7947998, -1008.3341064, 1016.4469604
3: -570.5770264, 758.5302124, -546.4636841, 726.1284180, -1296.7054443, 1304.9938965
4: -561.5471191, 831.9753418, -537.9462891, 797.6225586, -1359.1696777, 1369.9216309

Time for backsubstitution: 1.30 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1369226, upper bound: 1442.1368913
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1369226, upper bound: 1442.1372348
time: 0.99 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -569.1553955, 724.4523926, -652.5117798, 829.6059570, -1398.7613525, 1376.9641113
1: -450.1062012, 581.4873657, -515.7770386, 665.7797241, -1115.8857422, 1097.2642822
2: -391.1219788, 572.4326172, -447.9968872, 655.7226562, -1046.8446045, 1020.4295044
3: -531.4224854, 706.0717163, -608.9086304, 808.2564697, -1339.6789551, 1314.9802246
4: -523.2495117, 775.3944092, -599.4792480, 888.7031250, -1411.9525146, 1374.8735352

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361285, upper bound: 1442.1361495
time: 1.26 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361128, upper bound: 1442.1361474
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -610.8034058, 778.0040283, -652.5117798, 829.6059570, -1440.4088135, 1430.5156250
1: -482.7779541, 624.5186768, -515.7770386, 665.7797241, -1148.5576172, 1140.2955322
2: -419.5393982, 614.3937378, -447.9968872, 655.7226562, -1075.2620850, 1062.3906250
3: -570.5770264, 758.5302124, -608.9086304, 808.2564697, -1378.8334961, 1367.4388428
4: -561.5471191, 831.9753418, -599.4792480, 888.7031250, -1450.2501221, 1431.4543457

Time for backsubstitution: 1.31 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361285, upper bound: 1442.1361960
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361128, upper bound: 1442.1362177
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -790.9019775, 1006.4872437, -569.4039307, 726.3779907, -1517.2799072, 1575.8911133
1: -626.2216187, 810.0587158, -450.9070435, 583.0986328, -1209.3203125, 1260.9658203
2: -544.7996826, 796.9887085, -391.7252808, 574.5664062, -1119.3659668, 1188.7137451
3: -739.7652588, 982.8114624, -532.7806396, 707.8715210, -1447.6367188, 1515.5920410
4: -728.6350708, 1079.7222900, -524.1286621, 778.4970093, -1507.1318359, 1603.8507080

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1330917
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1330917
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -821.9379883, 1045.3026123, -569.4039307, 726.3779907, -1548.3159180, 1614.7065430
1: -650.1458130, 841.3367920, -450.9070435, 583.0986328, -1233.2443848, 1292.2438965
2: -565.6826782, 827.2041626, -391.7252808, 574.5664062, -1140.2490234, 1218.9294434
3: -768.1123047, 1020.8530884, -532.7806396, 707.8715210, -1475.9838867, 1553.6337891
4: -756.5951538, 1120.0638428, -524.1286621, 778.4970093, -1535.0919189, 1644.1922607

Time for backsubstitution: 1.32 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1332732
time: 1.07 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1332732
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -706.9984741, 901.3140869, -654.9114380, 832.8476562, -1539.8458252, 1556.2255859
1: -559.8485718, 725.2553101, -516.6654053, 667.6907959, -1227.5393066, 1241.9206543
2: -487.3349915, 714.4852905, -449.1707764, 657.2110596, -1144.5460205, 1163.6560059
3: -662.7079468, 879.7804565, -610.5296631, 809.9638672, -1472.6717529, 1490.3100586
4: -651.5615845, 968.1956787, -600.5786743, 890.2842407, -1541.8458252, 1568.7744141

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1360525
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1360526
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -729.5639648, 930.2368774, -654.9114380, 832.8476562, -1562.4114990, 1585.1483154
1: -577.4675293, 748.5848999, -516.6654053, 667.6907959, -1245.1582031, 1265.2502441
2: -502.6593628, 737.0874023, -449.1707764, 657.2110596, -1159.8703613, 1186.2579346
3: -683.7422485, 908.1420288, -610.5296631, 809.9638672, -1493.7060547, 1518.6716309
4: -672.1369019, 998.4097290, -600.5786743, 890.2842407, -1562.4211426, 1598.9884033

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1364044
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1364044
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -579.6847534, 736.5845947, -611.2731934, 777.6597900, -1357.3444824, 1347.8577881
1: -457.7289124, 591.3258667, -483.2408752, 624.5850220, -1082.3139648, 1074.5667725
2: -397.7966919, 581.4253540, -419.8180237, 614.1383057, -1011.9349976, 1001.2433472
3: -540.5828857, 718.1849976, -570.6395264, 758.7363281, -1299.3189697, 1288.8240967
4: -532.2545166, 787.2297363, -561.9980469, 831.6524658, -1363.9069824, 1349.2277832

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375250, upper bound: 1442.1374756
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375250, upper bound: 1442.1374757
time: 1.24 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -790.7902222, 1004.7871704, -611.2731934, 777.6597900, -1568.4499512, 1616.0603027
1: -625.2578735, 808.7488403, -483.2408752, 624.5850220, -1249.8427734, 1291.9897461
2: -544.0467529, 794.9827271, -419.8180237, 614.1383057, -1158.1849365, 1214.8005371
3: -738.5293579, 981.3143921, -570.6395264, 758.7363281, -1497.2651367, 1551.9536133
4: -727.5960693, 1076.4647217, -561.9980469, 831.6524658, -1559.2485352, 1638.4627686

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375250, upper bound: 1442.1374756
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375250, upper bound: 1442.1374757
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -552.7751465, 704.1228638, -804.4643555, 1023.2086792, -1575.9837646, 1508.5870361
1: -436.8860779, 565.1571045, -636.3490601, 823.5319214, -1260.4177246, 1201.5061035
2: -379.7437744, 556.2884521, -553.6569214, 809.6903687, -1189.4340820, 1109.9453125
3: -516.6292725, 686.1299438, -751.8293457, 999.3170776, -1515.9462891, 1437.9589844
4: -508.0301208, 753.2686768, -740.5961304, 1096.4323730, -1604.4625244, 1493.8647461

Time for backsubstitution: 1.33 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1369479, upper bound: 1442.1365770
time: 1.32 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1369479, upper bound: 1442.1365770
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -625.0759888, 793.4500122, -719.3660889, 917.1578369, -1542.2336426, 1512.8161621
1: -491.9536133, 635.5997314, -569.3789062, 738.0172119, -1229.9708252, 1204.9786377
2: -428.0535583, 625.4340210, -495.5730896, 726.6670532, -1154.7203369, 1121.0070801
3: -581.9815063, 770.9852905, -674.0648804, 895.3770142, -1477.3585205, 1445.0500488
4: -572.0442505, 846.9650879, -662.7022095, 984.2860107, -1556.3303223, 1509.6669922

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368916, upper bound: 1442.1365770
time: 0.79 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368916, upper bound: 1442.1365770
time: 1.01 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -780.5972290, 994.3472290, -615.3454590, 783.6099243, -1564.2071533, 1609.6926270
1: -617.9060669, 801.4954224, -486.5945435, 629.3319702, -1247.2377930, 1288.0899658
2: -536.2886963, 785.2061157, -422.7252197, 618.9642944, -1155.2529297, 1207.9313965
3: -728.8917236, 975.6419067, -574.7576904, 764.5104980, -1493.4020996, 1550.3995361
4: -719.4281006, 1062.0258789, -565.9726562, 838.2301636, -1557.6582031, 1627.9982910

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375223, upper bound: 1442.1375246
time: 1.21 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375223, upper bound: 1442.1375246
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -965.0853271, 1228.5307617, -615.3454590, 783.6099243, -1748.6953125, 1843.8760986
1: -763.8738403, 989.4804688, -486.5945435, 629.3319702, -1393.2054443, 1476.0749512
2: -663.9135132, 971.4710693, -422.7252197, 618.9642944, -1282.8778076, 1394.1961670
3: -901.9641724, 1201.9241943, -574.7576904, 764.5104980, -1666.4746094, 1776.6818848
4: -889.1920166, 1314.9461670, -565.9726562, 838.2301636, -1727.4221191, 1880.9185791

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375223, upper bound: 1442.1375246
time: 1.06 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375223, upper bound: 1442.1375246
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -780.5972290, 994.3472290, -818.9315796, 1041.9296875, -1822.5267334, 1813.2785645
1: -617.9060669, 801.4954224, -647.8930664, 838.6157837, -1456.5218506, 1449.3884277
2: -536.2886963, 785.2061157, -563.6307983, 824.5121460, -1360.8007812, 1348.8369141
3: -728.8917236, 975.6419067, -765.5197754, 1017.6923218, -1746.5839844, 1741.1276855
4: -719.4281006, 1062.0258789, -754.0483398, 1116.5321045, -1835.9600830, 1816.0740967

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375041, upper bound: 1442.1375041
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375041, upper bound: 1442.1375163
time: 1.24 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -965.0853271, 1228.5307617, -818.9315796, 1041.9296875, -2007.0150146, 2047.4622803
1: -763.8738403, 989.4804688, -647.8930664, 838.6157837, -1602.4896240, 1637.3735352
2: -663.9135132, 971.4710693, -563.6307983, 824.5121460, -1488.4256592, 1535.1015625
3: -901.9641724, 1201.9241943, -765.5197754, 1017.6923218, -1919.6564941, 1967.4438477
4: -889.1920166, 1314.9461670, -754.0483398, 1116.5321045, -2005.7241211, 2068.9943848

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375041, upper bound: 1442.1375041
time: 1.17 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1375041, upper bound: 1442.1375163
time: 1.10 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -634.1668091, 806.6979980, -584.7716064, 746.1163940, -1380.2827148, 1391.4696045
1: -500.6221313, 646.7952271, -462.8562012, 598.6146240, -1099.2366943, 1109.6513672
2: -435.1169434, 637.2324829, -402.1706238, 589.7974243, -1024.9143066, 1039.4030762
3: -591.8942871, 784.9700317, -547.0603027, 726.8192139, -1318.7135010, 1332.0302734
4: -581.9215698, 863.4533081, -538.0936890, 799.0889282, -1381.0104980, 1401.5469971

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361954, upper bound: 1442.1364428
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361954, upper bound: 1442.1365279
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -672.9708252, 859.6748657, -580.2546997, 740.7454834, -1413.7159424, 1439.9294434
1: -532.5321045, 689.2069092, -459.4139099, 594.2609863, -1126.7930908, 1148.6207275
2: -462.7770691, 679.6909180, -399.1734619, 585.5621338, -1048.3392334, 1078.8643799
3: -630.2377930, 836.3158569, -543.0509033, 721.5276489, -1351.7653809, 1379.3666992
4: -619.1112671, 921.4460449, -534.1317749, 793.3821411, -1412.4934082, 1455.5778809

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361954, upper bound: 1442.1364399
time: 0.82 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361954, upper bound: 1442.1365059
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -634.1668091, 806.6979980, -647.1181030, 823.6201172, -1457.7866211, 1453.8161621
1: -500.6221313, 646.7952271, -511.6114197, 660.6667480, -1161.2886963, 1158.4066162
2: -435.1169434, 637.2324829, -444.4057617, 651.0328979, -1086.1497803, 1081.6381836
3: -591.8942871, 784.9700317, -604.3847656, 801.9822388, -1393.8764648, 1389.3547363
4: -581.9215698, 863.4533081, -594.6277466, 882.4572144, -1464.3787842, 1458.0810547

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307443, upper bound: 1442.1339653
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307443, upper bound: 1442.1364332
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -672.9708252, 859.6748657, -643.8494263, 819.7515869, -1492.7224121, 1503.5242920
1: -532.5321045, 689.2069092, -509.1822815, 657.5438843, -1190.0759277, 1198.3891602
2: -462.7770691, 679.6909180, -442.2969971, 647.9874878, -1110.7645264, 1121.9877930
3: -630.2377930, 836.3158569, -601.5338745, 798.2066650, -1428.4442139, 1437.8496094
4: -619.1112671, 921.4460449, -591.8235474, 878.3515015, -1497.4627686, 1513.2695312

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307439, upper bound: 1442.1339646
time: 0.89 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1307439, upper bound: 1442.1364013
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -751.1771851, 953.9496460, -584.4492188, 744.7611694, -1495.9382324, 1538.3986816
1: -591.1172485, 763.3856812, -462.6269836, 597.6298828, -1188.7468262, 1226.0126953
2: -514.0568848, 751.9322510, -401.9172668, 588.6738892, -1102.7307129, 1153.8494873
3: -699.0958862, 925.9381104, -546.4501343, 725.6881104, -1424.7838135, 1472.3881836
4: -687.0231323, 1018.5568237, -537.7105103, 797.6400146, -1484.6628418, 1556.2670898

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361235, upper bound: 1442.1363632
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361235, upper bound: 1442.1363632
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -789.6867065, 1006.8889771, -580.2623901, 739.7507935, -1529.4375000, 1587.1513672
1: -622.8226318, 805.7930298, -459.4268494, 593.5744629, -1216.3968506, 1265.2198486
2: -541.5621948, 794.4070435, -399.1307068, 584.7159424, -1126.2780762, 1193.5377197
3: -737.3223267, 977.2608643, -542.7167358, 720.7618408, -1458.0842285, 1519.9775391
4: -724.0662231, 1076.5926514, -534.0261841, 792.3009644, -1516.3670654, 1610.6188965

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361024, upper bound: 1442.1363357
time: 0.81 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361024, upper bound: 1442.1363357
time: 1.01 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -751.1771851, 953.9496460, -654.7185059, 832.9525146, -1584.1293945, 1608.6677246
1: -591.1172485, 763.3856812, -517.8255615, 668.3597412, -1259.4770508, 1281.2111816
2: -514.0568848, 751.9322510, -449.7227173, 658.4416504, -1172.4985352, 1201.6550293
3: -699.0958862, 925.9381104, -611.3565674, 811.4793701, -1510.5748291, 1537.2944336
4: -687.0231323, 1018.5568237, -601.8016968, 892.4888916, -1579.5118408, 1620.3582764

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1305501, upper bound: 1442.1336357
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1305501, upper bound: 1442.1362124
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -789.6867065, 1006.8889771, -651.7498779, 829.4046631, -1619.0913086, 1658.6389160
1: -622.8226318, 805.7930298, -515.6206055, 665.5117188, -1288.3339844, 1321.4135742
2: -541.5621948, 794.4070435, -447.8067017, 655.6486816, -1197.2105713, 1242.2137451
3: -737.3223267, 977.2608643, -608.7501221, 808.0366211, -1545.3588867, 1586.0106201
4: -724.0662231, 1076.5926514, -599.2470093, 888.7144165, -1612.7805176, 1675.8395996

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

### Candidate
type: B, layer: 1, pos: 26

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 49

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1305498, upper bound: 1442.1336350
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1305498, upper bound: 1442.1361826
time: 0.96 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -606.8619995, 772.6406250, -581.1796265, 741.3933105, -1348.2553711, 1353.8203125
1: -478.8466187, 619.4959717, -459.6869812, 594.9623413, -1073.8089600, 1079.1829834
2: -416.3173828, 610.1140747, -399.3197021, 585.7528687, -1002.0702515, 1009.4337158
3: -566.6315918, 751.9107666, -543.4919434, 722.6522217, -1289.2838135, 1295.4027100
4: -557.2072144, 826.7517090, -534.8700562, 793.6293945, -1350.8364258, 1361.6215820

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367840, upper bound: 1442.1361943
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1366459, upper bound: 1442.1361953
time: 1.24 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -640.8618774, 818.2199097, -603.1608887, 769.2449341, -1410.1068115, 1421.3804932
1: -507.2322388, 656.8213501, -477.5698853, 617.9504395, -1125.1826172, 1134.3912354
2: -440.5025940, 646.7440186, -414.6264648, 608.0521240, -1048.5544434, 1061.3704834
3: -599.6865845, 797.5893555, -563.9782104, 750.7339478, -1350.4205322, 1361.5673828
4: -590.0802612, 876.5551758, -555.4552002, 823.7013550, -1413.7816162, 1432.0097656

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368612, upper bound: 1442.1367627
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367611, upper bound: 1442.1367611
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -606.8619995, 772.6406250, -736.6051636, 939.9682617, -1546.8302002, 1509.2457275
1: -478.8466187, 619.4959717, -582.3442993, 755.4843140, -1234.3309326, 1201.8403320
2: -416.3173828, 610.1140747, -506.9372864, 744.3112793, -1160.6285400, 1117.0512695
3: -566.6315918, 751.9107666, -689.7962036, 915.9344482, -1482.5660400, 1441.7069092
4: -557.2072144, 826.7517090, -677.8572998, 1007.9146118, -1565.1214600, 1504.6086426

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367812, upper bound: 1442.1361393
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1366459, upper bound: 1442.1361820
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -640.8618774, 818.2199097, -742.6681519, 947.0943604, -1587.9562988, 1560.8878174
1: -507.2322388, 656.8213501, -587.1197510, 761.1365356, -1268.3687744, 1243.9411621
2: -440.5025940, 646.7440186, -510.9956055, 749.7355347, -1190.2376709, 1157.7396240
3: -599.6865845, 797.5893555, -695.0194702, 922.9293823, -1522.6159668, 1492.6087646
4: -590.0802612, 876.5551758, -683.3251343, 1015.2970581, -1605.3771973, 1559.8798828

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 49

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368612, upper bound: 1442.1367416
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368007, upper bound: 1442.1367429
time: 1.30 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -716.4409180, 913.6683960, -572.4065552, 730.8249512, -1447.2658691, 1486.0749512
1: -565.7197266, 733.9050903, -453.1731873, 586.5167847, -1152.2365723, 1187.0782471
2: -492.5876160, 723.8386230, -393.5869446, 577.8439941, -1070.4316406, 1117.4254150
3: -670.7045898, 889.2468262, -535.8986816, 712.1765747, -1382.8811035, 1425.1455078
4: -658.3112793, 980.5217896, -526.9744873, 782.9265137, -1441.2377930, 1507.4962158

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1371324, upper bound: 1442.1371775
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1371324, upper bound: 1442.1372047
time: 1.03 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -878.6260376, 1118.0307617, -579.6435547, 740.9193115, -1619.5454102, 1697.6743164
1: -693.6141968, 899.0097656, -459.1412964, 594.6503296, -1288.2645264, 1358.1510010
2: -603.3898315, 883.5854492, -398.7500916, 585.9642944, -1189.3540039, 1282.3354492
3: -820.4733887, 1090.8843994, -543.1133423, 722.1426392, -1542.6159668, 1633.9978027
4: -807.4040527, 1195.5402832, -534.0069580, 793.9846802, -1601.3886719, 1729.5472412

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1371233, upper bound: 1442.1372768
time: 0.97 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1371233, upper bound: 1442.1373869
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -669.0029297, 855.7931519, -656.5040283, 835.3796997, -1504.3825684, 1512.2971191
1: -529.0385132, 687.2058716, -517.6673584, 669.2077026, -1198.2460938, 1204.8732910
2: -460.6263123, 678.7982788, -449.8865967, 658.9411621, -1119.5675049, 1128.6848145
3: -628.1516724, 832.7072754, -611.9802856, 811.9341431, -1440.0855713, 1444.6873779
4: -615.6333008, 920.0459595, -601.8223267, 892.6436768, -1508.2767334, 1521.8682861

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1365613, upper bound: 1442.1362881
time: 1.09 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1365213, upper bound: 1442.1362868
time: 0.97 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -839.5458374, 1070.4403076, -663.0924072, 844.7854004, -1684.3309326, 1733.5324707
1: -663.5917969, 860.7788086, -523.1721802, 676.8457642, -1340.4375000, 1383.9509277
2: -577.0628662, 846.5227661, -454.6401672, 666.5375366, -1243.6003418, 1301.1627197
3: -785.5648193, 1044.6417236, -618.6660156, 821.2318726, -1606.7966309, 1663.3077393
4: -772.3434448, 1145.6661377, -608.3364868, 902.9906616, -1675.3341064, 1754.0026855

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1366203, upper bound: 1442.1365883
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1365799, upper bound: 1442.1365799
time: 1.04 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.59 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1369226, upper bound: 1442.1368913
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1369226, upper bound: 1442.1368913
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1369226, upper bound: 1442.1368913
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1369226, upper bound: 1442.1372348
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361285, upper bound: 1442.1361495
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361128, upper bound: 1442.1361474
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361285, upper bound: 1442.1361960
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361128, upper bound: 1442.1362177
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1330917
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1330917
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1332732
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1303629, upper bound: 1442.1332732
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1360525
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1360526
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1364044
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1360083, upper bound: 1442.1364044
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375250, upper bound: 1442.1374756
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375250, upper bound: 1442.1374757
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375250, upper bound: 1442.1374756
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375250, upper bound: 1442.1374757
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1369479, upper bound: 1442.1365770
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1369479, upper bound: 1442.1365770
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1368916, upper bound: 1442.1365770
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1368916, upper bound: 1442.1365770
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375223, upper bound: 1442.1375246
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375223, upper bound: 1442.1375246
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375223, upper bound: 1442.1375246
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375223, upper bound: 1442.1375246
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375041, upper bound: 1442.1375041
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375041, upper bound: 1442.1375163
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375041, upper bound: 1442.1375041
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1375041, upper bound: 1442.1375163
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361954, upper bound: 1442.1364428
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361954, upper bound: 1442.1365279
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361954, upper bound: 1442.1364399
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361954, upper bound: 1442.1365059
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1307443, upper bound: 1442.1339653
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1307443, upper bound: 1442.1364332
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1307439, upper bound: 1442.1339646
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1307439, upper bound: 1442.1364013
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361235, upper bound: 1442.1363632
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361235, upper bound: 1442.1363632
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361024, upper bound: 1442.1363357
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1361024, upper bound: 1442.1363357
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1305501, upper bound: 1442.1336357
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1305501, upper bound: 1442.1362124
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1305498, upper bound: 1442.1336350
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1305498, upper bound: 1442.1361826
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1367840, upper bound: 1442.1361943
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1366459, upper bound: 1442.1361953
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1368612, upper bound: 1442.1367627
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1367611, upper bound: 1442.1367611
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1367812, upper bound: 1442.1361393
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1366459, upper bound: 1442.1361820
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1368612, upper bound: 1442.1367416
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1368007, upper bound: 1442.1367429
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1371324, upper bound: 1442.1371775
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1371324, upper bound: 1442.1372047
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1371233, upper bound: 1442.1372768
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1371233, upper bound: 1442.1373869
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1365613, upper bound: 1442.1362881
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1365213, upper bound: 1442.1362868
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1366203, upper bound: 1442.1365883
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.59
Output dim: 0, lower bound: -1442.1365799, upper bound: 1442.1365799

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -569.1553955, 724.4523926, -569.1553955, 724.4523926, -1293.6077881, 1293.6077881
1: -450.1062012, 581.4873657, -450.1062012, 581.4873657, -1031.5935059, 1031.5935059
2: -391.1219788, 572.4326172, -391.1219788, 572.4326172, -963.5545044, 963.5545044
3: -531.4224854, 706.0717163, -531.4224854, 706.0717163, -1237.4941406, 1237.4941406
4: -523.2495117, 775.3944092, -523.2495117, 775.3944092, -1298.6436768, 1298.6436768

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1363573, upper bound: 1442.1362569
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1363519, upper bound: 1442.1363040
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -569.1553955, 724.4523926, -797.1863403, 1014.1274414, -1583.2828369, 1521.6386719
1: -450.1062012, 581.4873657, -631.0876465, 816.2327881, -1266.3389893, 1212.5749512
2: -391.1219788, 572.4326172, -549.0272217, 802.9599609, -1194.0819092, 1121.4597168
3: -531.4224854, 706.0717163, -745.4263916, 990.3023682, -1521.7248535, 1451.4980469
4: -523.2495117, 775.3944092, -734.2783813, 1087.7778320, -1611.0273438, 1509.6727295

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1363573, upper bound: 1442.1362569
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1363519, upper bound: 1442.1363040
time: 0.96 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -610.8034058, 778.0040283, -569.1553955, 724.4523926, -1335.2553711, 1347.1594238
1: -482.7779541, 624.5186768, -450.1062012, 581.4873657, -1064.2653809, 1074.6245117
2: -419.5393982, 614.3937378, -391.1219788, 572.4326172, -991.9719238, 1005.5156250
3: -570.5770264, 758.5302124, -531.4224854, 706.0717163, -1276.6486816, 1289.9526367
4: -561.5471191, 831.9753418, -523.2495117, 775.3944092, -1336.9414062, 1355.2246094

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1364565, upper bound: 1442.1366282
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1364553, upper bound: 1442.1365203
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -610.8034058, 778.0040283, -797.1863403, 1014.1274414, -1624.9304199, 1575.1904297
1: -482.7779541, 624.5186768, -631.0876465, 816.2327881, -1299.0107422, 1255.6063232
2: -419.5393982, 614.3937378, -549.0272217, 802.9599609, -1222.4991455, 1163.4208984
3: -570.5770264, 758.5302124, -745.4263916, 990.3023682, -1560.8791504, 1503.9565430
4: -561.5471191, 831.9753418, -734.2783813, 1087.7778320, -1649.3249512, 1566.2536621

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1364565, upper bound: 1442.1366282
time: 1.11 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1364553, upper bound: 1442.1365203
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -568.4992676, 723.5439453, -648.1422119, 823.5639038, -1392.0629883, 1371.6861572
1: -449.5603638, 580.7530518, -512.1473999, 660.8873901, -1110.4477539, 1092.9001465
2: -390.6545410, 571.7141724, -444.8780212, 650.9395142, -1041.5939941, 1016.5921631
3: -530.7714844, 705.1686401, -604.5792236, 802.2393188, -1333.0107422, 1309.7478027
4: -522.6060791, 774.4179688, -595.1946411, 882.1995239, -1404.8055420, 1369.6124268

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361128, upper bound: 1442.1361474
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361128, upper bound: 1442.1361474
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -563.9653931, 718.0959473, -686.5898438, 875.9254761, -1439.8907471, 1404.6857910
1: -446.0945129, 576.3412476, -543.6965332, 702.7729492, -1148.8674316, 1120.0372314
2: -387.6390991, 567.4166260, -472.2153625, 692.8635254, -1080.5025635, 1039.6319580
3: -526.7208252, 699.8037720, -642.4405518, 852.9278564, -1379.6485596, 1342.2443848
4: -518.6072388, 768.6242065, -631.9031982, 939.4630737, -1458.0703125, 1400.5270996

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 6

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361128, upper bound: 1442.1361474
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361128, upper bound: 1442.1361474
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -610.1147461, 777.0610962, -648.1422119, 823.5639038, -1433.6783447, 1425.2031250
1: -482.2073059, 623.7542114, -512.1473999, 660.8873901, -1143.0947266, 1135.9013672
2: -419.0497437, 613.6521606, -444.8780212, 650.9395142, -1069.9892578, 1058.5300293
3: -569.8984375, 757.5891724, -604.5792236, 802.2393188, -1372.1376953, 1362.1684570
4: -560.8735352, 830.9696655, -595.1946411, 882.1995239, -1443.0729980, 1426.1640625

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361188, upper bound: 1442.1361960
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361188, upper bound: 1442.1361960
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -605.9006348, 772.0457153, -686.5898438, 875.9254761, -1481.8260498, 1458.6354980
1: -478.9971313, 619.6827393, -543.6965332, 702.7729492, -1181.7700195, 1163.3787842
2: -416.2591553, 609.6730347, -472.2153625, 692.8635254, -1109.1226807, 1081.8884277
3: -566.1616211, 752.6398926, -642.4405518, 852.9278564, -1419.0894775, 1395.0804443
4: -557.1901245, 825.6118164, -631.9031982, 939.4630737, -1496.6531982, 1457.5150146

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1358812, upper bound: 1442.1359226
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1355373, upper bound: 1442.1360074
time: 0.98 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -790.9019775, 1006.4872437, -541.3289185, 690.7130737, -1481.6149902, 1547.8160400
1: -626.2216187, 810.0587158, -428.4089050, 554.2188110, -1180.4404297, 1238.4676514
2: -544.7996826, 796.9887085, -372.3586426, 546.2640381, -1091.0637207, 1169.3470459
3: -739.7652588, 982.8114624, -506.5939636, 672.6951294, -1412.4604492, 1489.4051514
4: -728.6350708, 1079.7222900, -498.0998535, 740.1217651, -1468.7567139, 1577.8221436

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1302361, upper bound: 1442.1327333
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303464, upper bound: 1442.1330917
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -790.9019775, 1006.4872437, -592.3246460, 754.1802979, -1545.0821533, 1598.8118896
1: -626.2216187, 810.0587158, -468.1822510, 604.8382568, -1231.0598145, 1278.2409668
2: -544.7996826, 796.9887085, -406.7693787, 596.3756104, -1141.1750488, 1203.7579346
3: -739.7652588, 982.8114624, -553.4420166, 733.8547363, -1473.6199951, 1536.2534180
4: -728.6350708, 1079.7222900, -544.0882568, 808.3690796, -1537.0041504, 1623.8105469

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1302361, upper bound: 1442.1327333
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1303464, upper bound: 1442.1330917
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -821.9379883, 1045.3026123, -541.3289185, 690.7130737, -1512.6511230, 1586.6314697
1: -650.1458130, 841.3367920, -428.4089050, 554.2188110, -1204.3643799, 1269.7457275
2: -565.6826782, 827.2041626, -372.3586426, 546.2640381, -1111.9467773, 1199.5627441
3: -768.1123047, 1020.8530884, -506.5939636, 672.6951294, -1440.8073730, 1527.4465332
4: -756.5951538, 1120.0638428, -498.0998535, 740.1217651, -1496.7167969, 1618.1635742

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304347, upper bound: 1442.1332732
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304347, upper bound: 1442.1332732
time: 0.85 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -821.9379883, 1045.3026123, -592.3246460, 754.1802979, -1576.1180420, 1637.6271973
1: -650.1458130, 841.3367920, -468.1822510, 604.8382568, -1254.9837646, 1309.5190430
2: -565.6826782, 827.2041626, -406.7693787, 596.3756104, -1162.0583496, 1233.9735107
3: -768.1123047, 1020.8530884, -553.4420166, 733.8547363, -1501.9669189, 1574.2949219
4: -756.5951538, 1120.0638428, -544.0882568, 808.3690796, -1564.9642334, 1664.1519775

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 49

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304347, upper bound: 1442.1332732
time: 1.23 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1304347, upper bound: 1442.1332732
time: 0.94 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -706.9984741, 901.3140869, -625.5573120, 794.9502563, -1501.9484863, 1526.8713379
1: -559.8485718, 725.2553101, -493.0097656, 637.0538330, -1196.9023438, 1218.2651367
2: -487.3349915, 714.4852905, -428.8431396, 627.1203003, -1114.4552002, 1143.3283691
3: -662.7079468, 879.7804565, -582.9290161, 772.6624756, -1435.3703613, 1462.7094727
4: -651.5615845, 968.1956787, -573.1594238, 849.4346924, -1500.9960938, 1541.3551025

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1354365, upper bound: 1442.1353587
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1354935, upper bound: 1442.1354899
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -706.9984741, 901.3140869, -697.9887695, 887.2725830, -1594.2708740, 1599.3028564
1: -559.8485718, 725.2553101, -550.0432739, 710.6270752, -1270.4755859, 1275.2985840
2: -487.3349915, 714.4852905, -478.2152405, 700.0626831, -1187.3975830, 1192.7005615
3: -662.7079468, 879.7804565, -650.2662964, 862.0089111, -1524.7166748, 1530.0467529
4: -651.5615845, 968.1956787, -639.4726562, 948.5807495, -1600.1422119, 1607.6683350

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1354365, upper bound: 1442.1353587
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1354935, upper bound: 1442.1354899
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -729.5639648, 930.2368774, -625.5573120, 794.9502563, -1524.5141602, 1555.7940674
1: -577.4675293, 748.5848999, -493.0097656, 637.0538330, -1214.5212402, 1241.5947266
2: -502.6593628, 737.0874023, -428.8431396, 627.1203003, -1129.7796631, 1165.9304199
3: -683.7422485, 908.1420288, -582.9290161, 772.6624756, -1456.4047852, 1491.0710449
4: -672.1369019, 998.4097290, -573.1594238, 849.4346924, -1521.5715332, 1571.5690918

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1356518, upper bound: 1442.1357603
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1356468, upper bound: 1442.1357894
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -729.5639648, 930.2368774, -697.9887695, 887.2725830, -1616.8365479, 1628.2255859
1: -577.4675293, 748.5848999, -550.0432739, 710.6270752, -1288.0943604, 1298.6281738
2: -502.6593628, 737.0874023, -478.2152405, 700.0626831, -1202.7220459, 1215.3026123
3: -683.7422485, 908.1420288, -650.2662964, 862.0089111, -1545.7510986, 1558.4083252
4: -672.1369019, 998.4097290, -639.4726562, 948.5807495, -1620.7176514, 1637.8823242

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1356518, upper bound: 1442.1357603
time: 1.18 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1356468, upper bound: 1442.1357894
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -579.6847534, 736.5845947, -586.9890137, 746.6256104, -1326.3103027, 1323.5733643
1: -457.7289124, 591.3258667, -463.6170044, 599.3323364, -1057.0612793, 1054.9428711
2: -397.7966919, 581.4253540, -402.9101562, 589.4132690, -987.2099609, 984.3355103
3: -540.5828857, 718.1849976, -547.7460327, 727.9478149, -1268.5307617, 1265.9310303
4: -532.2545166, 787.2297363, -539.2005005, 798.0526123, -1330.3071289, 1326.4301758

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367578, upper bound: 1442.1368126
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367447, upper bound: 1442.1366922
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -579.6847534, 736.5845947, -637.8111572, 810.7407227, -1390.4255371, 1374.3953857
1: -457.7289124, 591.3258667, -503.6018372, 650.5854492, -1108.3143311, 1094.9277344
2: -397.7966919, 581.4253540, -437.5944519, 640.3605957, -1038.1572266, 1019.0197144
3: -540.5828857, 718.1849976, -595.0585938, 789.8191528, -1330.4020996, 1313.2434082
4: -532.2545166, 787.2297363, -585.5843506, 867.5877686, -1399.8422852, 1372.8140869

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367578, upper bound: 1442.1368126
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367447, upper bound: 1442.1366922
time: 0.99 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -790.7902222, 1004.7871704, -586.9890137, 746.6256104, -1537.4157715, 1591.7761230
1: -625.2578735, 808.7488403, -463.6170044, 599.3323364, -1224.5902100, 1272.3658447
2: -544.0467529, 794.9827271, -402.9101562, 589.4132690, -1133.4597168, 1197.8927002
3: -738.5293579, 981.3143921, -547.7460327, 727.9478149, -1466.4771729, 1529.0604248
4: -727.5960693, 1076.4647217, -539.2005005, 798.0526123, -1525.6485596, 1615.6652832

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367219, upper bound: 1442.1368126
time: 1.05 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367207, upper bound: 1442.1366922
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -790.7902222, 1004.7871704, -637.8111572, 810.7407227, -1601.5310059, 1642.5982666
1: -625.2578735, 808.7488403, -503.6018372, 650.5854492, -1275.8431396, 1312.3507080
2: -544.0467529, 794.9827271, -437.5944519, 640.3605957, -1184.4073486, 1232.5770264
3: -738.5293579, 981.3143921, -595.0585938, 789.8191528, -1528.3483887, 1576.3729248
4: -727.5960693, 1076.4647217, -585.5843506, 867.5877686, -1595.1837158, 1662.0490723

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367219, upper bound: 1442.1368126
time: 1.18 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367207, upper bound: 1442.1366922
time: 1.20 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -552.7751465, 704.1228638, -793.7358398, 1010.0802002, -1562.8553467, 1497.8586426
1: -436.8860779, 565.1571045, -627.9987793, 812.9373169, -1249.8232422, 1193.1558838
2: -379.7437744, 556.2884521, -546.4149780, 799.4077759, -1179.1514893, 1102.7033691
3: -516.6292725, 686.1299438, -742.1018066, 986.4236450, -1503.0528564, 1428.2316895
4: -508.0301208, 753.2686768, -730.9244385, 1082.5654297, -1590.5955811, 1484.1931152

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1371096, upper bound: 1442.1369453
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1370712, upper bound: 1442.1368581
time: 1.05 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -552.7751465, 704.1228638, -739.9794922, 943.3677368, -1496.1428223, 1444.1022949
1: -436.8860779, 565.1571045, -583.8916626, 757.8438110, -1194.7297363, 1149.0488281
2: -379.7437744, 556.2884521, -508.6620789, 746.9345093, -1126.6782227, 1064.9505615
3: -516.6292725, 686.1299438, -692.6694336, 918.6558228, -1435.2851562, 1378.7991943
4: -508.0301208, 753.2686768, -679.7151489, 1011.7458496, -1519.7760010, 1432.9836426

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 46

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1371096, upper bound: 1442.1369453
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1370712, upper bound: 1442.1368581
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -625.0759888, 793.4500122, -705.2725220, 898.9683838, -1524.0439453, 1498.7225342
1: -491.9536133, 635.5997314, -558.1499634, 723.3712158, -1215.3248291, 1193.7497559
2: -428.0535583, 625.4340210, -485.8134460, 712.2172241, -1140.2706299, 1111.2474365
3: -581.9815063, 770.9852905, -660.7437744, 877.5820312, -1459.5634766, 1431.7290039
4: -572.0442505, 846.9650879, -649.6054077, 964.6857300, -1536.7299805, 1496.5700684

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367589, upper bound: 1442.1364176
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368721, upper bound: 1442.1364961
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368737, upper bound: 1442.1365592
time: 1.28 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -625.0759888, 793.4500122, -891.5172119, 1135.8417969, -1760.9177246, 1684.9670410
1: -491.9536133, 635.5997314, -706.1334839, 914.7810059, -1406.7346191, 1341.7331543
2: -428.0535583, 625.4340210, -613.5911865, 898.4533691, -1326.5069580, 1239.0251465
3: -581.9815063, 770.9852905, -834.1096191, 1111.6088867, -1693.5902100, 1605.0948486
4: -572.0442505, 846.9650879, -821.7904053, 1215.9639893, -1788.0083008, 1668.7552490

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367589, upper bound: 1442.1364194
time: 1.28 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368721, upper bound: 1442.1364961
time: 1.08 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368737, upper bound: 1442.1365592
time: 0.82 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -780.5972290, 994.3472290, -589.5667725, 750.7024536, -1531.2995605, 1583.9140625
1: -617.9060669, 801.4954224, -465.7408447, 602.5693970, -1220.4754639, 1267.2363281
2: -536.2886963, 785.2061157, -404.7640686, 592.7388916, -1129.0275879, 1189.9699707
3: -728.8917236, 975.6419067, -550.4979248, 731.8818359, -1460.7733154, 1526.1396484
4: -719.4281006, 1062.0258789, -541.8021240, 802.6371460, -1522.0651855, 1603.8280029

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368284, upper bound: 1442.1369589
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368117, upper bound: 1442.1367423
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -780.5972290, 994.3472290, -645.3641357, 820.9956055, -1601.5927734, 1639.7114258
1: -617.9060669, 801.4954224, -509.8055420, 658.9353027, -1276.8413086, 1311.3009033
2: -536.2886963, 785.2061157, -442.9554138, 648.5541382, -1184.8427734, 1228.1614990
3: -728.8917236, 975.6419067, -602.4577026, 800.0009155, -1528.8925781, 1578.0992432
4: -719.4281006, 1062.0258789, -592.8767090, 878.6969604, -1598.1248779, 1654.9024658

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368284, upper bound: 1442.1369589
time: 0.98 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1368117, upper bound: 1442.1367423
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -965.0853271, 1228.5307617, -589.5667725, 750.7024536, -1715.7878418, 1818.0974121
1: -763.8738403, 989.4804688, -465.7408447, 602.5693970, -1366.4432373, 1455.2213135
2: -663.9135132, 971.4710693, -404.7640686, 592.7388916, -1256.6523438, 1376.2347412
3: -901.9641724, 1201.9241943, -550.4979248, 731.8818359, -1633.8459473, 1752.4219971
4: -889.1920166, 1314.9461670, -541.8021240, 802.6371460, -1691.8291016, 1856.7482910

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367244, upper bound: 1442.1368898
time: 0.96 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367231, upper bound: 1442.1367210
time: 1.14 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -965.0853271, 1228.5307617, -645.3641357, 820.9956055, -1786.0809326, 1873.8947754
1: -763.8738403, 989.4804688, -509.8055420, 658.9353027, -1422.8090820, 1499.2860107
2: -663.9135132, 971.4710693, -442.9554138, 648.5541382, -1312.4676514, 1414.4261475
3: -901.9641724, 1201.9241943, -602.4577026, 800.0009155, -1701.9650879, 1804.3817139
4: -889.1920166, 1314.9461670, -592.8767090, 878.6969604, -1767.8889160, 1907.8228760

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367244, upper bound: 1442.1368898
time: 1.15 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367231, upper bound: 1442.1367210
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -780.5972290, 994.3472290, -807.5957031, 1026.9141846, -1807.5113525, 1801.9426270
1: -617.9060669, 801.4954224, -638.7462769, 826.5715942, -1444.4776611, 1440.2415771
2: -536.2886963, 785.2061157, -555.7216797, 812.5584717, -1348.8471680, 1340.9277344
3: -728.8917236, 975.6419067, -754.6206665, 1003.0495605, -1731.9409180, 1730.1754150
4: -719.4281006, 1062.0258789, -743.3699341, 1100.2539062, -1819.6820068, 1805.3953857

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367927, upper bound: 1442.1369531
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367769, upper bound: 1442.1367231
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -780.5972290, 994.3472290, -741.5993042, 945.8302002, -1726.4272461, 1735.9465332
1: -617.9060669, 801.4954224, -585.8828735, 759.9152222, -1377.8212891, 1387.3782959
2: -536.2886963, 785.2061157, -510.0263062, 749.3134766, -1285.6021729, 1295.2324219
3: -728.8917236, 975.6419067, -694.3344727, 920.8809814, -1649.7725830, 1669.9763184
4: -719.4281006, 1062.0258789, -681.7730103, 1014.9357910, -1734.3637695, 1743.7987061

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367927, upper bound: 1442.1369531
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367769, upper bound: 1442.1367358
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -965.0853271, 1228.5307617, -807.5957031, 1026.9141846, -1991.9995117, 2036.1260986
1: -763.8738403, 989.4804688, -638.7462769, 826.5715942, -1590.4453125, 1628.2266846
2: -663.9135132, 971.4710693, -555.7216797, 812.5584717, -1476.4719238, 1527.1926270
3: -901.9641724, 1201.9241943, -754.6206665, 1003.0495605, -1905.0136719, 1956.5446777
4: -889.1920166, 1314.9461670, -743.3699341, 1100.2539062, -1989.4459229, 2058.3161621

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367056, upper bound: 1442.1368804
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367043, upper bound: 1442.1367043
time: 1.19 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -965.0853271, 1228.5307617, -741.5993042, 945.8302002, -1910.9155273, 1970.1301270
1: -763.8738403, 989.4804688, -585.8828735, 759.9152222, -1523.7890625, 1575.3632812
2: -663.9135132, 971.4710693, -510.0263062, 749.3134766, -1413.2270508, 1481.4971924
3: -901.9641724, 1201.9241943, -694.3344727, 920.8809814, -1822.8452148, 1896.2586670
4: -889.1920166, 1314.9461670, -681.7730103, 1014.9357910, -1904.1276855, 1996.7191162

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 49

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367056, upper bound: 1442.1368851
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1367043, upper bound: 1442.1367151
time: 1.06 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -634.1668091, 806.6979980, -558.3981323, 713.4005737, -1347.5673828, 1365.0959473
1: -500.6221313, 646.7952271, -442.0394592, 572.1254883, -1072.7475586, 1088.8347168
2: -435.1169434, 637.2324829, -384.1871948, 564.0808716, -999.1976929, 1021.4196777
3: -591.8942871, 784.9700317, -522.9218750, 694.5599365, -1286.4542236, 1307.8918457
4: -581.9215698, 863.4533081, -514.0005493, 764.4040527, -1346.3256836, 1377.4534912

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1324880, upper bound: 1442.1294374
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1324880, upper bound: 1442.1364428
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -634.1668091, 806.6979980, -669.0694580, 851.2661743, -1485.4326172, 1475.7674561
1: -500.6221313, 646.7952271, -527.4647217, 681.6071167, -1182.2291260, 1174.2600098
2: -435.1169434, 637.2324829, -458.7009888, 671.3104858, -1106.4274902, 1095.9334717
3: -591.8942871, 784.9700317, -623.8507080, 826.9270630, -1418.8212891, 1408.8203125
4: -581.9215698, 863.4533081, -612.9685059, 909.3377686, -1491.2592773, 1476.4215088

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1324880, upper bound: 1442.1294374
time: 1.08 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1324880, upper bound: 1442.1365279
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -672.9708252, 859.6748657, -554.0648804, 708.2939453, -1381.2647705, 1413.7397461
1: -532.5321045, 689.2069092, -438.7358704, 567.9829712, -1100.5151367, 1127.9426270
2: -462.7770691, 679.6909180, -381.3084717, 560.0385742, -1022.8154907, 1060.9992676
3: -630.2377930, 836.3158569, -519.0903931, 689.5254517, -1319.7631836, 1355.4062500
4: -619.1112671, 921.4460449, -510.2182922, 758.9554443, -1378.0666504, 1431.6643066

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1330699, upper bound: 1442.1303563
time: 0.94 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1330699, upper bound: 1442.1364399
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -672.9708252, 859.6748657, -664.9860229, 846.4785156, -1519.4493408, 1524.6608887
1: -532.5321045, 689.2069092, -524.3442383, 677.7197876, -1210.2519531, 1213.5511475
2: -462.7770691, 679.6909180, -455.9856567, 667.5065918, -1130.2833252, 1135.6765137
3: -630.2377930, 836.3158569, -620.2470093, 822.2031860, -1452.4407959, 1456.5628662
4: -619.1112671, 921.4460449, -609.4148560, 904.2180176, -1523.3292236, 1530.8608398

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1330699, upper bound: 1442.1303563
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1330699, upper bound: 1442.1303563
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -634.1668091, 806.6979980, -618.1671753, 787.1115723, -1421.2781982, 1424.8651123
1: -500.6221313, 646.7952271, -488.4817200, 630.9740601, -1131.5960693, 1135.2769775
2: -435.1169434, 637.2324829, -424.4437866, 622.1121216, -1057.2288818, 1061.6762695
3: -591.8942871, 784.9700317, -577.4899292, 765.8117676, -1357.7059326, 1362.4599609
4: -581.9215698, 863.4533081, -567.7404785, 843.3607788, -1425.2821045, 1431.1938477

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1299750, upper bound: 1442.1291310
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1299750, upper bound: 1442.1339653
time: 1.07 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -634.1668091, 806.6979980, -747.8726807, 950.5761108, -1584.7427979, 1554.5705566
1: -500.6221313, 646.7952271, -589.0722046, 760.7600098, -1261.3820801, 1235.8674316
2: -435.1169434, 637.2324829, -512.1409302, 749.5523071, -1184.6691895, 1149.3734131
3: -591.8942871, 784.9700317, -696.5556641, 922.9223633, -1514.8165283, 1481.5255127
4: -581.9215698, 863.4533081, -684.6231689, 1015.5466309, -1597.4681396, 1548.0762939

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1299750, upper bound: 1442.1293825
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1299750, upper bound: 1442.1293825
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -672.9708252, 859.6748657, -615.0520020, 783.4871826, -1456.4580078, 1474.7268066
1: -532.5321045, 689.2069092, -486.1709290, 628.0389404, -1160.5710449, 1175.3778076
2: -462.7770691, 679.6909180, -422.4356384, 619.2434082, -1082.0203857, 1102.1264648
3: -630.2377930, 836.3158569, -574.7944336, 762.2688599, -1392.5063477, 1411.1101074
4: -619.1112671, 921.4460449, -565.0944824, 839.4881592, -1458.5993652, 1486.5405273

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1299882, upper bound: 1442.1299882
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1299882, upper bound: 1442.1299882
time: 1.02 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -672.9708252, 859.6748657, -744.8023682, 947.0877075, -1620.0583496, 1604.4772949
1: -532.5321045, 689.2069092, -586.8101196, 757.9217529, -1290.4538574, 1276.0170898
2: -462.7770691, 679.6909180, -510.1749878, 746.7763672, -1209.5534668, 1189.8657227
3: -630.2377930, 836.3158569, -693.9343262, 919.4999390, -1549.7377930, 1530.2502441
4: -619.1112671, 921.4460449, -682.0535278, 1011.8149414, -1630.9261475, 1603.4995117

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 31

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 18

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1299882, upper bound: 1442.1302884
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1299882, upper bound: 1442.1302884
time: 1.18 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -751.1771851, 953.9496460, -558.3909912, 713.3919067, -1464.5688477, 1512.3405762
1: -591.1172485, 763.3856812, -442.0338440, 572.1186523, -1163.2358398, 1205.4195557
2: -514.0568848, 751.9322510, -384.1823120, 564.0740356, -1078.1308594, 1136.1143799
3: -699.0958862, 925.9381104, -522.9154663, 694.5516357, -1393.6473389, 1448.8535156
4: -687.0231323, 1018.5568237, -513.9941406, 764.3948364, -1451.4179688, 1532.5507812

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297893, upper bound: 1442.1252856
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361235, upper bound: 1442.1363632
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -751.1771851, 953.9496460, -669.0694580, 851.2661743, -1602.4429932, 1623.0187988
1: -591.1172485, 763.3856812, -527.4647217, 681.6071167, -1272.7241211, 1290.8503418
2: -514.0568848, 751.9322510, -458.7009888, 671.3104858, -1185.3674316, 1210.6333008
3: -699.0958862, 925.9381104, -623.8507080, 826.9270630, -1526.0229492, 1549.7885742
4: -687.0231323, 1018.5568237, -612.9685059, 909.3377686, -1596.3608398, 1631.5251465

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1297893, upper bound: 1442.1253328
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1361235, upper bound: 1442.1363632
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -789.6867065, 1006.8889771, -554.0578613, 708.2854004, -1497.9721680, 1560.9467773
1: -622.8226318, 805.7930298, -438.7302246, 567.9763794, -1190.7990723, 1244.5231934
2: -541.5621948, 794.4070435, -381.3036804, 560.0317993, -1101.5938721, 1175.7106934
3: -737.3223267, 977.2608643, -519.0839233, 689.5173340, -1426.8395996, 1496.3446045
4: -724.0662231, 1076.5926514, -510.2120361, 758.9464111, -1483.0125732, 1586.8044434

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1358047, upper bound: 1442.1357690
time: 1.10 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1358047, upper bound: 1442.1363357
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -789.6867065, 1006.8889771, -664.9860229, 846.4785156, -1636.1652832, 1671.8750000
1: -622.8226318, 805.7930298, -524.3442383, 677.7197876, -1300.5423584, 1330.1372070
2: -541.5621948, 794.4070435, -455.9856567, 667.5065918, -1209.0682373, 1250.3927002
3: -737.3223267, 977.2608643, -620.2470093, 822.2031860, -1559.5255127, 1597.5075684
4: -724.0662231, 1076.5926514, -609.4148560, 904.2180176, -1628.2840576, 1686.0074463

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 10

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1358047, upper bound: 1442.1357690
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1358047, upper bound: 1442.1363357
time: 0.97 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -751.1771851, 953.9496460, -618.1300049, 787.0660400, -1538.2430420, 1572.0795898
1: -591.1172485, 763.3856812, -488.4526978, 630.9384155, -1222.0552979, 1251.8383789
2: -514.0568848, 751.9322510, -424.4184265, 622.0766602, -1136.1334229, 1176.3507080
3: -699.0958862, 925.9381104, -577.4561157, 765.7686768, -1464.8643799, 1503.3942871
4: -687.0231323, 1018.5568237, -567.7074585, 843.3125000, -1530.3354492, 1586.2642822

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1289528, upper bound: 1442.1252784
time: 1.23 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1305474, upper bound: 1442.1336357
time: 1.08 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -751.1771851, 953.9496460, -747.8588867, 950.5597534, -1701.7368164, 1701.8084717
1: -591.1172485, 763.3856812, -589.0614624, 760.7470703, -1351.8642578, 1352.4471436
2: -514.0568848, 751.9322510, -512.1314697, 749.5394287, -1263.5963135, 1264.0637207
3: -699.0958862, 925.9381104, -696.5432129, 922.9067993, -1622.0023193, 1622.4812012
4: -687.0231323, 1018.5568237, -684.6110229, 1015.5291748, -1702.5521240, 1703.1678467

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 3

### Candidate
type: A, layer: 1, pos: 26

### Candidate
type: A, layer: 1, pos: 42

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1289528, upper bound: 1442.1287498
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1442.1305474, upper bound: 1442.1362124
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -789.6867065, 1006.8889771, -615.0150757, 783.4414673, -1573.1281738, 1621.9040527
1: -622.8226318, 805.7930298, -486.1419373, 628.0032349, -1250.8259277, 1291.9349365
2: -541.5621948, 794.4070435, -422.4103088, 619.2076416, -1160.7697754, 1216.8173828
3: -737.3223267, 977.2608643, -574.7606812, 762.2256470, -1499.5479736, 1552.0211182
4: -724.0662231, 1076.5926514, -565.0614014, 839.4395752, -1563.5057373, 1641.6540527

Time for backsubstitution: 1.56 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.63 + 416.66 = 420.29 seconds
