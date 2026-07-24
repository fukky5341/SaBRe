## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_2.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
Time budget: 420 seconds
Split limit: 100
Threshold: 1757.072574941339


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-395.7757874, 1606.8325195, -395.7757874, 1606.8325195, -2002.6080322, 2002.6081543)
1: (-485.0866699, 1794.6639404, -485.0866699, 1794.6639404, -2279.7504883, 2279.7504883)
2: (-554.9915771, 1824.4812012, -554.9915771, 1824.4812012, -2379.4726562, 2379.4726562)
3: (-788.0418701, 1987.3795166, -788.0418701, 1987.3795166, -2775.4213867, 2775.4213867)
4: (-931.7708130, 1855.1873779, -931.7708130, 1855.1873779, -2786.9582520, 2786.9582520)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.48 + 2.04 = 3.52 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1757.1252887, upper bound: 1757.1252887

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 10

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1145273, upper bound: 1757.1052763
time: 1.08 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044230, upper bound: 1757.1044230
time: 0.84 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.05 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.05
Output dim: 0, lower bound: -1757.1145273, upper bound: 1757.1052763
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.05
Output dim: 0, lower bound: -1757.1044230, upper bound: 1757.1044230

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -356.5766602, 1445.0728760, -371.2193604, 1505.7791748, -1862.3558350, 1816.2921143
1: -437.0072937, 1613.3231201, -454.9303589, 1681.3889160, -2118.3957520, 2068.2534180
2: -500.5809937, 1640.3117676, -520.9175415, 1709.1297607, -2209.7106934, 2161.2292480
3: -710.3735962, 1789.6778564, -739.4672852, 1863.7824707, -2574.1557617, 2529.1450195
4: -841.4490967, 1668.5886230, -875.3663330, 1738.4481201, -2579.8972168, 2543.9548340

Time for backsubstitution: 1.34 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043455, upper bound: 1757.1043455
time: 0.67 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043455, upper bound: 1757.1044186
time: 0.68 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -457.7469177, 1887.2629395, -371.2248840, 1513.6586914, -1971.4056396, 2258.4877930
1: -559.7200317, 2107.2399902, -454.8751221, 1690.3209229, -2250.0405273, 2562.1149902
2: -649.0897827, 2136.3076172, -521.4410400, 1717.9735107, -2367.0632324, 2657.7485352
3: -917.2413330, 2344.1276855, -739.9755249, 1871.7624512, -2789.0039062, 3084.1032715
4: -1100.8392334, 2173.5678711, -876.6570435, 1746.9129639, -2847.7519531, 3050.2248535

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0614236, upper bound: 1757.0372694
time: 0.64 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1044230, upper bound: 1757.1044230
time: 0.66 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.76 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 0, lower bound: -1757.1043455, upper bound: 1757.1043455
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 0, lower bound: -1757.1043455, upper bound: 1757.1044186
NS_A2_B1, status: Status.VERIFIED, split count: 2, time: 2.76
Output dim: 0, lower bound: -1757.0614236, upper bound: 1757.0372694
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.76
Output dim: 0, lower bound: -1757.1044230, upper bound: 1757.1044230

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -356.5766602, 1445.0728760, -356.5766602, 1445.0728760, -1801.6495361, 1801.6495361
1: -437.0072937, 1613.3231201, -437.0072937, 1613.3231201, -2050.3303223, 2050.3303223
2: -500.5809937, 1640.3117676, -500.5809937, 1640.3117676, -2140.8925781, 2140.8928223
3: -710.3735962, 1789.6778564, -710.3735962, 1789.6778564, -2500.0515137, 2500.0515137
4: -841.4490967, 1668.5886230, -841.4490967, 1668.5886230, -2510.0375977, 2510.0375977

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0929643, upper bound: 1757.0635633
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1145273, upper bound: 1757.1052402
time: 0.69 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -356.5766602, 1445.0728760, -457.6540527, 1886.8894043, -2243.4658203, 1902.7269287
1: -437.0072937, 1613.3231201, -559.6082153, 2106.8259277, -2543.8332520, 2172.9311523
2: -500.5809937, 1640.3117676, -648.9575806, 2135.8837891, -2636.4645996, 2289.2690430
3: -710.3735962, 1789.6778564, -917.0573730, 2343.6613770, -3054.0349121, 2706.7351074
4: -841.4490967, 1668.5886230, -1100.6107178, 2173.1391602, -3014.5883789, 2769.1992188

Time for backsubstitution: 1.35 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0929643, upper bound: 1757.0635633
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1145273, upper bound: 1757.1052763
time: 0.77 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -452.7701416, 1867.0562744, -368.8015747, 1504.0627441, -1956.8326416, 2235.8571777
1: -553.6854248, 2084.7058105, -451.9244080, 1679.6166992, -2233.3015137, 2536.6296387
2: -642.0420532, 2113.4272461, -518.0854492, 1707.0693359, -2349.1113281, 2631.5124512
3: -907.3236694, 2318.9487305, -735.2094116, 1859.8853760, -2767.2089844, 3054.1582031
4: -1088.7954102, 2150.3833008, -871.0351562, 1735.8483887, -2824.6435547, 3021.4184570

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0965724, upper bound: 1757.0946814
time: 0.76 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0946814, upper bound: 1757.0946814
time: 0.72 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.96 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.0929643, upper bound: 1757.0635633
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.1145273, upper bound: 1757.1052402
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.0929643, upper bound: 1757.0635633
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.1145273, upper bound: 1757.1052763
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.0965724, upper bound: 1757.0946814
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 2.96
Output dim: 0, lower bound: -1757.0946814, upper bound: 1757.0946814

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -314.7524109, 1277.9980469, -339.9203796, 1376.4552002, -1691.2075195, 1617.9183350
1: -385.6337891, 1426.8695068, -416.7227783, 1536.6734619, -1922.3070068, 1843.5922852
2: -441.0567932, 1450.1931152, -476.9831848, 1562.5854492, -2003.6420898, 1927.1762695
3: -626.3041992, 1580.1624756, -677.0227051, 1704.9387207, -2331.2429199, 2257.1850586
4: -740.3444824, 1474.6290283, -801.4036865, 1589.9177246, -2330.2619629, 2276.0327148

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1165374
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -352.7528076, 1429.8717041, -353.9714050, 1434.7081299, -1787.4609375, 1783.8430176
1: -432.3590393, 1596.3819580, -433.8390503, 1601.7735596, -2034.1324463, 2030.2208252
2: -495.2670288, 1623.0344238, -496.9551392, 1628.5278320, -2123.7946777, 2119.9895020
3: -702.8384399, 1770.8116455, -705.2367554, 1776.8187256, -2479.6569824, 2476.0483398
4: -832.5136719, 1651.0700684, -835.3566284, 1656.6358643, -2489.1491699, 2486.4267578

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1109456, upper bound: 1757.1165632
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1106829, upper bound: 1757.1106829
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -314.7524109, 1277.9980469, -443.8089294, 1830.7186279, -2145.4709473, 1721.8070068
1: -385.6337891, 1426.8695068, -542.5246582, 2044.0688477, -2429.7023926, 1969.3939209
2: -441.0567932, 1450.1931152, -629.5624390, 2072.1047363, -2513.1616211, 2079.7556152
3: -626.3041992, 1580.1624756, -889.4066162, 2274.6928711, -2900.9968262, 2469.5688477
4: -740.3444824, 1474.6290283, -1068.3784180, 2108.4841309, -2848.8286133, 2543.0068359

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0813842, upper bound: 1757.0511321
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0716818, upper bound: 1757.0360599
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -352.7528076, 1429.8717041, -452.6768494, 1866.6801758, -2219.4331055, 1882.5484619
1: -432.3590393, 1596.3819580, -553.5733032, 2084.2893066, -2516.6477051, 2149.9550781
2: -495.2670288, 1623.0344238, -641.9092407, 2113.0014648, -2608.2683105, 2264.9436035
3: -702.8384399, 1770.8116455, -907.1387939, 2318.4792480, -3021.3173828, 2677.9504395
4: -832.5136719, 1651.0700684, -1088.5656738, 2149.9521484, -2982.4658203, 2739.6357422

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1017333, upper bound: 1757.0964243
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1058169, upper bound: 1757.0946450
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -420.5928650, 1736.3977051, -359.5854492, 1466.9517822, -1887.5446777, 2095.9826660
1: -514.5111694, 1939.6591797, -440.6658020, 1638.1665039, -2152.6777344, 2380.3249512
2: -595.4641113, 1966.3552246, -505.2915344, 1665.0169678, -2260.4809570, 2471.6467285
3: -842.4025269, 2153.4938965, -716.9827881, 1813.9501953, -2656.3527832, 2870.4765625
4: -1008.0108643, 1999.4211426, -849.5573730, 1693.0642090, -2701.0747070, 2848.9785156

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0963128, upper bound: 1757.0945847
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0963128, upper bound: 1757.0946814
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -445.8331909, 1837.3640137, -366.7913208, 1495.6477051, -1941.4805908, 2204.1552734
1: -545.3447266, 2051.5466309, -449.4855042, 1670.1864014, -2215.5312500, 2501.0319824
2: -632.1391602, 2080.0646973, -515.2437744, 1697.6159668, -2329.7551270, 2595.3085938
3: -893.4653931, 2282.0004883, -731.1838379, 1849.4750977, -2742.9404297, 3013.1843262
4: -1071.7197266, 2116.3774414, -866.1734619, 1726.2407227, -2797.9602051, 2982.5507812

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 10

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0944489, upper bound: 1757.0945847
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0944489, upper bound: 1757.0946814
time: 0.66 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 2.89 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1165374
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.1109456, upper bound: 1757.1165632
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.1106829, upper bound: 1757.1106829
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0813842, upper bound: 1757.0511321
NS_A1_B2_A1_B2, status: Status.VERIFIED, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0716818, upper bound: 1757.0360599
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.1017333, upper bound: 1757.0964243
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.1058169, upper bound: 1757.0946450
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0963128, upper bound: 1757.0945847
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0963128, upper bound: 1757.0946814
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0944489, upper bound: 1757.0945847
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 2.89
Output dim: 0, lower bound: -1757.0944489, upper bound: 1757.0946814

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -305.5162964, 1240.8395996, -296.1171570, 1203.4157715, -1508.9318848, 1536.9565430
1: -374.3565369, 1385.3933105, -363.3607788, 1344.2764893, -1718.6330566, 1748.7541504
2: -428.3149414, 1408.0085449, -414.2772217, 1366.2524414, -1794.5673828, 1822.2857666
3: -608.0607910, 1534.3875732, -589.7463379, 1487.1267090, -2095.1875000, 2124.1337891
4: -719.2200317, 1431.6661377, -694.1582031, 1389.6391602, -2108.8581543, 2125.8242188

Time for backsubstitution: 1.36 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -312.9614868, 1270.6021729, -337.0799255, 1364.6414795, -1677.6030273, 1607.6820068
1: -383.4582214, 1418.5880127, -413.2757568, 1523.4416504, -1906.8999023, 1831.8637695
2: -438.5548706, 1441.8253174, -473.0045471, 1549.2425537, -1987.7973633, 1914.8297119
3: -622.7549438, 1571.0299072, -671.3787842, 1690.2877197, -2313.0427246, 2242.4086914
4: -736.1218872, 1466.1414795, -794.6588745, 1576.3179932, -2312.4399414, 2260.8000488

Time for backsubstitution: 1.38 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0582156, upper bound: 1757.0854867
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -342.8920593, 1390.2570801, -310.6676636, 1263.7501221, -1606.6420898, 1700.9246826
1: -420.2868347, 1552.1630859, -381.0783997, 1411.6885986, -1831.9752197, 1933.2414551
2: -481.5502625, 1578.0948486, -434.8324585, 1434.5958252, -1916.1461182, 2012.9272461
3: -683.2750854, 1721.9652100, -618.8957520, 1561.4095459, -2244.6845703, 2340.8608398
4: -809.6408081, 1605.3092041, -729.1145630, 1458.5157471, -2268.1564941, 2334.4238281

Time for backsubstitution: 1.37 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1081712, upper bound: 1757.1102394
time: 0.65 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -350.9931641, 1422.5258789, -351.0003052, 1422.3206787, -1773.3134766, 1773.5261230
1: -430.2149048, 1588.1575928, -430.2168274, 1587.9046631, -2018.1195068, 2018.3743896
2: -492.7920227, 1614.7390137, -492.7782593, 1614.5341797, -2107.3261719, 2107.5173340
3: -699.3253784, 1761.7321777, -699.3073120, 1761.5272217, -2460.8525391, 2461.0395508
4: -828.3157349, 1642.6568604, -828.2799072, 1642.4438477, -2470.7595215, 2470.9367676

Time for backsubstitution: 1.39 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1074923, upper bound: 1757.1026508
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -299.7273254, 1217.3317871, -430.8540649, 1775.8489990, -2075.5761719, 1648.1857910
1: -367.2577515, 1358.9992676, -526.0120239, 1982.6040039, -2349.8613281, 1885.0112305
2: -420.1833496, 1381.3266602, -611.6985474, 2009.2030029, -2429.3862305, 1993.0251465
3: -596.3986206, 1504.9160156, -862.0715332, 2208.7536621, -2805.1523438, 2366.9870605
4: -704.8568726, 1404.6832275, -1037.8847656, 2045.7440186, -2750.6008301, 2442.5676270

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0427900, upper bound: 1757.0426295
time: 0.69 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0776368, upper bound: 1757.0433232
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -342.8920593, 1390.2570801, -419.8938293, 1733.6481934, -2076.5400391, 1810.1508789
1: -420.2868347, 1552.1630859, -513.6599121, 1936.6042480, -2356.8911133, 2065.8229980
2: -481.5502625, 1578.0948486, -594.5081177, 1963.2020264, -2444.7519531, 2172.6030273
3: -683.2750854, 1721.9652100, -841.0315552, 2150.1069336, -2833.3820801, 2562.9963379
4: -809.6408081, 1605.3092041, -1006.3929443, 1996.2548828, -2805.8952637, 2611.7021484

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0999657, upper bound: 1757.0939629
time: 0.67 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1017159, upper bound: 1757.0963371
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -350.9931641, 1422.5258789, -445.7468872, 1837.0157471, -2188.0087891, 1868.2727051
1: -430.2149048, 1588.1575928, -545.2409668, 2051.1599121, -2481.3747559, 2133.3979492
2: -492.7920227, 1614.7390137, -632.0169678, 2079.6682129, -2572.4602051, 2246.7556152
3: -699.3253784, 1761.7321777, -893.2947998, 2281.5676270, -2980.8930664, 2655.0268555
4: -828.3157349, 1642.6568604, -1071.5086670, 2115.9787598, -2944.2944336, 2714.1652832

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1056983, upper bound: 1757.0943661
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1020959, upper bound: 1757.0945602
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -420.5928650, 1736.3977051, -342.1352539, 1387.3868408, -1807.9797363, 2078.5329590
1: -514.5111694, 1939.6591797, -419.3670349, 1549.0183105, -2063.5295410, 2359.0261230
2: -595.4641113, 1966.3552246, -480.4833374, 1574.8337402, -2170.2978516, 2446.8386230
3: -842.4025269, 2153.4938965, -681.7869873, 1718.3395996, -2560.7421875, 2835.2807617
4: -1008.0108643, 1999.4211426, -807.8242798, 1602.0373535, -2610.0471191, 2807.2453613

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0936795
time: 0.66 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0955394, upper bound: 1757.0943352
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -420.5928650, 1736.3977051, -444.2421265, 1832.9116211, -2253.5043945, 2180.6398926
1: -514.5111694, 1939.6591797, -543.1940308, 2046.6008301, -2561.1120605, 2482.8530273
2: -595.4641113, 1966.3552246, -630.1496582, 2074.6374512, -2670.1015625, 2596.5046387
3: -842.4025269, 2153.4938965, -890.3460083, 2276.8747559, -3119.2773438, 3043.8398438
4: -1008.0108643, 1999.4211426, -1068.9468994, 2110.9748535, -3118.9855957, 3068.3676758

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 23

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 30

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0936850
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0955394, upper bound: 1757.0944475
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -445.8331909, 1837.3640137, -350.1552734, 1419.3455811, -1865.1785889, 2187.5192871
1: -545.3447266, 2051.5466309, -429.1974792, 1584.6760254, -2130.0207520, 2480.7441406
2: -632.1391602, 2080.0646973, -491.6131592, 1611.1276855, -2243.2668457, 2571.6777344
3: -893.4653931, 2282.0004883, -697.6809082, 1757.7150879, -2651.1801758, 2979.6813965
4: -1071.7197266, 2116.3774414, -826.3066406, 1639.0408936, -2710.7604980, 2942.6840820

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0621563, upper bound: 1757.0359613
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0343377, upper bound: 1757.0343377
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -445.8331909, 1837.3640137, -445.8584595, 1838.0943604, -2283.9274902, 2283.2224121
1: -545.3447266, 2051.5466309, -545.3385620, 2052.4045410, -2597.7490234, 2596.8850098
2: -632.1391602, 2080.0646973, -632.2102051, 2080.7358398, -2712.8750000, 2712.2749023
3: -893.4653931, 2282.0004883, -893.5246582, 2282.9143066, -3176.3796387, 3175.5251465
4: -1071.7197266, 2116.3774414, -1071.9210205, 2117.1904297, -3188.9101562, 3188.2978516

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0621563, upper bound: 1757.0359613
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0343377, upper bound: 1757.0343377
time: 0.63 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 2.93 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0582156, upper bound: 1757.0854867
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.1081712, upper bound: 1757.1102394
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.1074923, upper bound: 1757.1026508
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
NS_A1_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0427900, upper bound: 1757.0426295
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0776368, upper bound: 1757.0433232
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0999657, upper bound: 1757.0939629
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.1017159, upper bound: 1757.0963371
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.1056983, upper bound: 1757.0943661
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.1020959, upper bound: 1757.0945602
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0936795
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0955394, upper bound: 1757.0943352
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0936850
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0955394, upper bound: 1757.0944475
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0621563, upper bound: 1757.0359613
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0343377, upper bound: 1757.0343377
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0621563, upper bound: 1757.0359613
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 2.93
Output dim: 0, lower bound: -1757.0343377, upper bound: 1757.0343377

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -272.6042480, 1111.4747314, -296.1171570, 1203.4157715, -1476.0200195, 1407.5915527
1: -334.2237854, 1241.4110107, -363.3607788, 1344.2764893, -1678.5002441, 1604.7717285
2: -381.0059814, 1261.2246094, -414.2772217, 1366.2524414, -1747.2583008, 1675.5018311
3: -542.2633057, 1371.0507812, -589.7463379, 1487.1267090, -2029.3900146, 1960.7971191
4: -637.6555786, 1281.7025146, -694.1582031, 1389.6391602, -2027.2946777, 1975.8607178

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1164502
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1165374
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -311.7570190, 1265.6446533, -296.1171570, 1203.4157715, -1515.1727295, 1561.7617188
1: -381.9933777, 1413.0380859, -363.3607788, 1344.2764893, -1726.2695312, 1776.3989258
2: -436.8719788, 1436.2128906, -414.2772217, 1366.2524414, -1803.1242676, 1850.4901123
3: -620.3704224, 1564.9118652, -589.7463379, 1487.1267090, -2107.4970703, 2154.6582031
4: -733.2883911, 1460.4486084, -694.1582031, 1389.6391602, -2122.9267578, 2154.6066895

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1164502
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1165374
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -306.8064575, 1236.9456787, -328.0059509, 1328.4783936, -1635.2849121, 1564.9514160
1: -375.5833130, 1381.5285645, -402.2465210, 1483.1811523, -1858.7644043, 1783.7749023
2: -429.5268860, 1403.6560059, -460.0093079, 1508.2561035, -1937.7829590, 1863.6651611
3: -609.1093750, 1533.5759277, -653.3298340, 1644.6927490, -2253.8022461, 2186.9057617
4: -721.4624634, 1428.5528564, -772.3366089, 1534.6732178, -2256.1347656, 2200.8894043

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0582156, upper bound: 1757.0854867
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0582156, upper bound: 1757.0854867
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -308.7390442, 1252.9986572, -335.1853943, 1356.7490234, -1665.4880371, 1588.1840820
1: -378.3237610, 1398.9570312, -410.9712524, 1514.6379395, -1892.9616699, 1809.9282227
2: -432.6511841, 1421.8994141, -470.3675537, 1540.3150635, -1972.9663086, 1892.2669678
3: -614.3208618, 1549.3441162, -667.5976562, 1680.5698242, -2294.8903809, 2216.9418945
4: -726.1154175, 1445.9741211, -790.1995850, 1567.2680664, -2293.3835449, 2236.1735840

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -334.0846252, 1355.9155273, -306.4111938, 1247.1842041, -1581.2687988, 1662.3266602
1: -409.5643616, 1513.7418213, -375.8803406, 1393.1300049, -1802.6943359, 1889.6221924
2: -469.3659363, 1539.0684814, -428.9467773, 1415.7695312, -1885.1354980, 1968.0151367
3: -666.1113892, 1679.8684082, -610.6079712, 1541.1982422, -2207.3090820, 2290.4763184
4: -789.5351562, 1565.8653564, -719.4127197, 1439.6606445, -2229.1958008, 2285.2778320

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -364.5095825, 1477.6470947, -307.8875427, 1252.4083252, -1616.9177246, 1785.5345459
1: -446.9963684, 1649.8228760, -377.6715698, 1399.0207520, -1846.0170898, 2027.4943848
2: -511.9837341, 1677.5418701, -430.9226379, 1421.7504883, -1933.7342529, 2108.4645996
3: -726.5914307, 1830.8710938, -613.3203735, 1547.0090332, -2273.6005859, 2444.1914062
4: -860.5737915, 1706.9511719, -722.3775024, 1445.4515381, -2306.0251465, 2429.3286133

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -342.3549194, 1388.7806396, -346.6248779, 1405.2835693, -1747.6381836, 1735.4055176
1: -419.6886292, 1550.3975830, -424.9064636, 1568.8382568, -1988.5266113, 1975.3040771
2: -480.8171997, 1576.3961182, -486.7361145, 1595.1795654, -2075.9968262, 2063.1320801
3: -682.4650879, 1720.4185791, -690.8009644, 1740.6284180, -2423.0935059, 2411.2189941
4: -808.5219727, 1603.9654541, -818.2977905, 1622.8994141, -2431.4213867, 2422.2631836

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
time: 0.64 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -372.6387634, 1509.7955322, -348.2245178, 1410.8449707, -1783.4837646, 1858.0200195
1: -456.9413147, 1685.6937256, -426.8068848, 1575.1096191, -2032.0509033, 2112.5004883
2: -523.2628784, 1714.0583496, -488.8564453, 1601.5622559, -2124.8251953, 2202.9147949
3: -742.6359253, 1870.5782471, -693.7055054, 1747.2542725, -2489.8901367, 2564.2836914
4: -879.2671509, 1744.1877441, -821.5169678, 1629.3121338, -2508.5791016, 2565.7043457

Time for backsubstitution: 1.40 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -295.4109802, 1199.3037109, -429.9007568, 1771.8751221, -2067.2861328, 1629.2044678
1: -362.0229797, 1338.8834229, -524.8253174, 1978.1604004, -2340.1833496, 1863.7087402
2: -414.1443176, 1360.9234619, -610.3620605, 2004.6878662, -2418.8320312, 1971.2851562
3: -587.7935791, 1482.6755371, -860.1400146, 2203.9147949, -2791.7080078, 2342.8154297
4: -694.5849609, 1384.0321045, -1035.6717529, 2041.1646729, -2735.7492676, 2419.7036133

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0755939, upper bound: 1757.0416238
time: 0.76 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0755939, upper bound: 1757.0433232
time: 0.77 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -334.0846252, 1355.9155273, -416.1751099, 1720.0622559, -2054.1469727, 1772.0905762
1: -409.5643616, 1513.7418213, -509.0836792, 1921.3770752, -2330.9411621, 2022.8254395
2: -469.3659363, 1539.0684814, -589.4460449, 1947.5854492, -2416.9514160, 2128.5146484
3: -666.1113892, 1679.8684082, -833.8807983, 2133.4125977, -2799.5239258, 2513.7492676
4: -789.5351562, 1565.8653564, -998.2853394, 1980.4028320, -2769.9379883, 2564.1506348

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -364.5095825, 1477.6470947, -415.1767883, 1713.6027832, -2078.1123047, 1892.8237305
1: -446.9963684, 1649.8228760, -507.9197998, 1914.2021484, -2361.1982422, 2157.7426758
2: -511.9837341, 1677.5418701, -587.8978882, 1940.6174316, -2452.6010742, 2265.4396973
3: -726.5914307, 1830.8710938, -831.5453491, 2125.4062500, -2851.9975586, 2662.4165039
4: -860.5737915, 1706.9511719, -995.0518188, 1973.5129395, -2834.0866699, 2702.0029297

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 23

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0944415, upper bound: 1757.0923174
time: 1.12 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1012209, upper bound: 1757.0955368
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -342.3549194, 1388.7806396, -442.1566772, 1823.8242188, -2166.1791992, 1830.9372559
1: -419.6886292, 1550.3975830, -540.8420410, 2036.3802490, -2456.0688477, 2091.2397461
2: -480.8171997, 1576.3961182, -627.1105347, 2064.5339355, -2545.3510742, 2203.5065918
3: -682.4650879, 1720.4185791, -886.4033813, 2265.3244629, -2947.7893066, 2606.8215332
4: -808.5219727, 1603.9654541, -1063.6241455, 2100.6477051, -2909.1696777, 2667.5891113

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 30

## Relational analysis of NS_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0860896, upper bound: 1757.0528544
time: 0.68 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0999657, upper bound: 1757.0943408
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -372.6387634, 1509.7955322, -440.9219360, 1816.3114014, -2188.9501953, 1950.7175293
1: -456.9413147, 1685.6937256, -539.3720703, 2028.0289307, -2484.9702148, 2225.0654297
2: -523.2628784, 1714.0583496, -625.2587280, 2056.3601074, -2579.6230469, 2339.3171387
3: -742.6359253, 1870.5782471, -883.5599976, 2256.0671387, -2998.7031250, 2754.1381836
4: -879.2671509, 1744.1877441, -1059.8828125, 2092.5019531, -2971.7690430, 2804.0700684

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0827961, upper bound: 1757.0526688
time: 0.72 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1020959, upper bound: 1757.0945392
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -413.9170227, 1709.2200928, -341.0546875, 1382.7980957, -1796.7149658, 2050.2744141
1: -506.4060364, 1909.3297119, -418.0574036, 1543.8829346, -2050.2890625, 2327.3872070
2: -586.0026245, 1935.4637451, -478.9678040, 1569.6392822, -2155.6413574, 2414.4309082
3: -829.1879272, 2119.7893066, -679.6377563, 1712.7128906, -2541.8996582, 2799.4270020
4: -992.0071411, 1968.1402588, -805.2593994, 1596.7941895, -2588.8012695, 2773.3996582

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0944555
time: 0.70 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0944555
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -418.5886841, 1728.0035400, -341.3080444, 1384.1033936, -1802.6921387, 2069.3115234
1: -512.0783081, 1930.2691650, -418.3627930, 1545.3494873, -2057.4277344, 2348.6315918
2: -592.6408691, 1956.8780518, -479.3136292, 1571.1064453, -2163.7470703, 2436.1916504
3: -838.3815918, 2143.0317383, -680.1620483, 1714.2401123, -2552.6215820, 2823.1933594
4: -1003.1513672, 1989.7790527, -805.8431396, 1598.2559814, -2601.4072266, 2795.6220703

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0956611, upper bound: 1757.1012410
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0956611, upper bound: 1757.1012410
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -413.9170227, 1709.2200928, -443.1513977, 1828.4537354, -2242.3708496, 2152.3715820
1: -506.4060364, 1909.3297119, -541.8543701, 2041.6053467, -2548.0114746, 2451.1840820
2: -586.0026245, 1935.4637451, -628.6166992, 2069.5720215, -2655.5744629, 2564.0800781
3: -829.1879272, 2119.7893066, -888.1821899, 2271.4270020, -3100.6145020, 3007.9714355
4: -992.0071411, 1968.1402588, -1066.4084473, 2105.8566895, -3097.8637695, 3034.5488281

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0936850
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0936850
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -418.5886841, 1728.0035400, -443.5432129, 1830.0454102, -2248.6340332, 2171.5468750
1: -512.0783081, 1930.2691650, -542.3400879, 2043.3952637, -2555.4736328, 2472.6093750
2: -592.6408691, 1956.8780518, -629.1741943, 2071.3825684, -2664.0234375, 2586.0522461
3: -838.3815918, 2143.0317383, -888.9470825, 2273.3115234, -3111.6931152, 3031.9787598
4: -1003.1513672, 1989.7790527, -1067.2857666, 2107.6613770, -3110.8125000, 3057.0649414

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0955409, upper bound: 1757.0944475
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0955409, upper bound: 1757.0944475
time: 0.69 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 2.98 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1164502
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1165374
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1164502
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1165374
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0582156, upper bound: 1757.0854867
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0582156, upper bound: 1757.0854867
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1128202, upper bound: 1757.1106562
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0755939, upper bound: 1757.0416238
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0755939, upper bound: 1757.0433232
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0944415, upper bound: 1757.0923174
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1012209, upper bound: 1757.0955368
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0860896, upper bound: 1757.0528544
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0999657, upper bound: 1757.0943408
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0827961, upper bound: 1757.0526688
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.1020959, upper bound: 1757.0945392
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0944555
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0944555
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0956611, upper bound: 1757.1012410
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0956611, upper bound: 1757.1012410
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0936850
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0924367, upper bound: 1757.0936850
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0955409, upper bound: 1757.0944475
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 2.98
Output dim: 0, lower bound: -1757.0955409, upper bound: 1757.0944475

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -272.6042480, 1111.4747314, -272.2157288, 1109.8139648, -1382.4182129, 1383.6901855
1: -334.2237854, 1241.4110107, -333.7474670, 1239.5623779, -1573.7860107, 1575.1584473
2: -381.0059814, 1261.2246094, -380.4518127, 1259.3488770, -1640.3546143, 1641.6762695
3: -542.2633057, 1371.0507812, -541.4631348, 1368.9881592, -1911.2514648, 1912.5139160
4: -637.6555786, 1281.7025146, -636.7003784, 1279.8031006, -1917.4584961, 1918.4028320

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0940500, upper bound: 1757.1106456
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1174104, upper bound: 1757.1172664
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -272.6042480, 1111.4747314, -308.8294983, 1256.3496094, -1528.9538574, 1420.3040771
1: -334.2237854, 1241.4110107, -378.8326416, 1403.4211426, -1737.6448975, 1620.2435303
2: -381.0059814, 1261.2246094, -432.2922058, 1426.1948242, -1807.2008057, 1693.5168457
3: -542.2633057, 1371.0507812, -615.2606201, 1552.2734375, -2094.5366211, 1986.3112793
4: -637.6555786, 1281.7025146, -724.8505859, 1449.9963379, -2087.6518555, 2006.5529785

Time for backsubstitution: 1.41 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0940500, upper bound: 1757.1106456
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1174104, upper bound: 1757.1175706
time: 0.80 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -311.7570190, 1265.6446533, -272.2157288, 1109.8139648, -1421.5708008, 1537.8603516
1: -381.9933777, 1413.0380859, -333.7474670, 1239.5623779, -1621.5552979, 1746.7855225
2: -436.8719788, 1436.2128906, -380.4518127, 1259.3488770, -1696.2205811, 1816.6645508
3: -620.3704224, 1564.9118652, -541.4631348, 1368.9881592, -1989.3586426, 2106.3750000
4: -733.2883911, 1460.4486084, -636.7003784, 1279.8031006, -2013.0915527, 2097.1489258

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1045515, upper bound: 1757.1134242
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1045515, upper bound: 1757.1164502
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -311.7570190, 1265.6446533, -308.8294983, 1256.3496094, -1568.1065674, 1574.4741211
1: -381.9933777, 1413.0380859, -378.8326416, 1403.4211426, -1785.4143066, 1791.8706055
2: -436.8719788, 1436.2128906, -432.2922058, 1426.1948242, -1863.0666504, 1868.5051270
3: -620.3704224, 1564.9118652, -615.2606201, 1552.2734375, -2172.6437988, 2180.1721191
4: -733.2883911, 1460.4486084, -724.8505859, 1449.9963379, -2183.2839355, 2185.2988281

Time for backsubstitution: 1.42 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1045515, upper bound: 1757.1134242
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1165374
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -306.8064575, 1236.9456787, -303.2882385, 1231.6737061, -1538.4802246, 1540.2338867
1: -375.5833130, 1381.5285645, -371.6848450, 1375.2067871, -1750.7900391, 1753.2130127
2: -429.5268860, 1403.6560059, -424.6935730, 1397.6848145, -1827.2116699, 1828.3496094
3: -609.1093750, 1533.5759277, -603.4017944, 1522.2044678, -2131.3139648, 2136.9777832
4: -721.4624634, 1428.5528564, -712.2661133, 1421.2578125, -2142.7199707, 2140.8188477

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0420178, upper bound: 1757.0693308
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0568420, upper bound: 1757.0851416
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0579947, upper bound: 1757.0845757
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -306.8064575, 1236.9456787, -340.4590454, 1380.4127197, -1687.2192383, 1577.4046631
1: -375.5833130, 1381.5285645, -417.4032288, 1541.2288818, -1916.8122559, 1798.9316406
2: -429.5268860, 1403.6560059, -477.7629395, 1566.9912109, -1996.5180664, 1881.4189453
3: -609.1093750, 1533.5759277, -678.3824463, 1708.7834473, -2317.8928223, 2211.9584961
4: -721.4624634, 1428.5528564, -802.5859375, 1594.1118164, -2315.5742188, 2231.1386719

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0420178, upper bound: 1757.0693308
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0568420, upper bound: 1757.0851416
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0579947, upper bound: 1757.0845757
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -308.7390442, 1252.9986572, -309.8907166, 1257.8829346, -1566.6219482, 1562.8894043
1: -378.3237610, 1398.9570312, -379.7238770, 1404.3829346, -1782.7066650, 1778.6809082
2: -432.6511841, 1421.8994141, -434.2645874, 1427.4250488, -1860.0761719, 1856.1638184
3: -614.3208618, 1549.3441162, -616.6444092, 1555.3499756, -2169.6708984, 2165.9885254
4: -726.1154175, 1445.9741211, -728.8696289, 1451.5528564, -2177.6682129, 2174.8435059

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0624251, upper bound: 1757.0886251
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1047296, upper bound: 1757.1017128
time: 1.05 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -308.7390442, 1252.9986572, -347.7300720, 1408.9136963, -1717.6527100, 1600.7285156
1: -378.3237610, 1398.9570312, -426.2337341, 1572.9533691, -1951.2770996, 1825.1907959
2: -432.6511841, 1421.8994141, -488.2292480, 1599.3488770, -2032.0000000, 1910.1286621
3: -614.3208618, 1549.3441162, -692.8020020, 1744.9897461, -2359.3105469, 2242.1459961
4: -726.1154175, 1445.9741211, -820.5954590, 1627.0676270, -2353.1831055, 2266.5695801

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0624251, upper bound: 1757.0886251
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1047296, upper bound: 1757.1017128
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -334.0846252, 1355.9155273, -302.4146423, 1231.6340332, -1565.7186279, 1658.3300781
1: -409.5643616, 1513.7418213, -371.0033875, 1375.7783203, -1785.3426514, 1884.7452393
2: -469.3659363, 1539.0684814, -423.4243774, 1398.1044922, -1867.4704590, 1962.4926758
3: -666.1113892, 1679.8684082, -602.8259277, 1522.2109375, -2188.3222656, 2282.6936035
4: -789.5351562, 1565.8653564, -710.2945557, 1421.9650879, -2211.5002441, 2276.1599121

Time for backsubstitution: 1.43 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1081712, upper bound: 1757.1102394
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1081712, upper bound: 1757.1102394
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -334.0846252, 1355.9155273, -331.6198425, 1348.4567871, -1682.5413818, 1687.5350342
1: -409.5643616, 1513.7418213, -406.9620056, 1506.3532715, -1915.9176025, 1920.7038574
2: -469.3659363, 1539.0684814, -464.4823608, 1531.0493164, -2000.4152832, 2003.5506592
3: -666.1113892, 1679.8684082, -660.9517822, 1667.2908936, -2333.4023438, 2340.8198242
4: -789.5351562, 1565.8653564, -778.8173828, 1557.4371338, -2346.9721680, 2344.6826172

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1081712, upper bound: 1757.1102394
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1081712, upper bound: 1757.1102394
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -364.5095825, 1477.6470947, -302.4146423, 1231.6340332, -1596.1433105, 1780.0617676
1: -446.9963684, 1649.8228760, -371.0033875, 1375.7783203, -1822.7746582, 2020.8262939
2: -511.9837341, 1677.5418701, -423.4243774, 1398.1044922, -1910.0882568, 2100.9663086
3: -726.5914307, 1830.8710938, -602.8259277, 1522.2109375, -2248.8022461, 2433.6967773
4: -860.5737915, 1706.9511719, -710.2945557, 1421.9650879, -2282.5388184, 2417.2456055

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
time: 0.74 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -364.5095825, 1477.6470947, -331.6198425, 1348.4567871, -1712.9660645, 1809.2666016
1: -446.9963684, 1649.8228760, -406.9620056, 1506.3532715, -1953.3496094, 2056.7849121
2: -511.9837341, 1677.5418701, -464.4823608, 1531.0493164, -2043.0327148, 2142.0241699
3: -726.5914307, 1830.8710938, -660.9517822, 1667.2908936, -2393.8823242, 2491.8227539
4: -860.5737915, 1706.9511719, -778.8173828, 1557.4371338, -2418.0109863, 2485.7685547

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
time: 0.82 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -342.3549194, 1388.7806396, -342.5917358, 1389.5111084, -1731.8659668, 1731.3723145
1: -419.6886292, 1550.3975830, -419.9836426, 1551.1816406, -1970.8699951, 1970.3812256
2: -480.8171997, 1576.3961182, -481.1403809, 1577.2651367, -2058.0822754, 2057.5358887
3: -682.4650879, 1720.4185791, -682.9180298, 1721.3345947, -2403.7998047, 2403.3359375
4: -808.5219727, 1603.9654541, -809.0411377, 1604.8242188, -2413.3461914, 2413.0061035

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1074923, upper bound: 1757.1026508
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1074923, upper bound: 1757.1026508
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -342.3549194, 1388.7806396, -372.8626404, 1510.5230713, -1852.8779297, 1761.6433105
1: -419.6886292, 1550.3975830, -457.2156067, 1686.4685059, -2106.1572266, 2007.6129150
2: -480.8171997, 1576.3961182, -523.5686646, 1714.9011230, -2195.7182617, 2099.9643555
3: -682.4650879, 1720.4185791, -743.0819702, 1871.5242920, -2553.9892578, 2463.5004883
4: -808.5219727, 1603.9654541, -879.7874146, 1745.0615234, -2553.5834961, 2483.7529297

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1074923, upper bound: 1757.1026508
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1074923, upper bound: 1757.1026508
time: 0.65 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -372.6387634, 1509.7955322, -342.5917358, 1389.5111084, -1762.1499023, 1852.3872070
1: -456.9413147, 1685.6937256, -419.9836426, 1551.1816406, -2008.1228027, 2105.6765137
2: -523.2628784, 1714.0583496, -481.1403809, 1577.2651367, -2100.5280762, 2195.1987305
3: -742.6359253, 1870.5782471, -682.9180298, 1721.3345947, -2463.9704590, 2553.4963379
4: -879.2671509, 1744.1877441, -809.0411377, 1604.8242188, -2484.0913086, 2553.2282715

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -372.6387634, 1509.7955322, -372.8626404, 1510.5230713, -1883.1618652, 1882.6580811
1: -456.9413147, 1685.6937256, -457.2156067, 1686.4685059, -2143.4099121, 2142.9094238
2: -523.2628784, 1714.0583496, -523.5686646, 1714.9011230, -2238.1640625, 2237.6269531
3: -742.6359253, 1870.5782471, -743.0819702, 1871.5242920, -2614.1601562, 2613.6601562
4: -879.2671509, 1744.1877441, -879.7874146, 1745.0615234, -2624.3286133, 2623.9750977

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
time: 0.74 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -295.4109802, 1199.3037109, -419.4445190, 1727.5482178, -2022.9592285, 1618.7481689
1: -362.0229797, 1338.8834229, -511.4711304, 1928.6611328, -2290.6840820, 1850.3544922
2: -414.1443176, 1360.9234619, -595.4890747, 1954.5865479, -2368.7309570, 1956.4125977
3: -587.7935791, 1482.6755371, -838.4092407, 2150.3488770, -2738.1420898, 2321.0847168
4: -694.5849609, 1384.0321045, -1010.7836914, 1990.4708252, -2685.0551758, 2394.8146973

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0754006, upper bound: 1757.0416133
time: 0.65 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0755938, upper bound: 1757.0416238
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -295.4109802, 1199.3037109, -428.8708801, 1767.4667969, -2062.8776855, 1628.1745605
1: -362.0229797, 1338.8834229, -523.5390015, 1973.2327881, -2335.2558594, 1862.4221191
2: -414.1443176, 1360.9234619, -608.9116211, 1999.6789551, -2413.8225098, 1969.8350830
3: -587.7935791, 1482.6755371, -858.0338745, 2198.5661621, -2786.3596191, 2340.7094727
4: -694.5849609, 1384.0321045, -1033.2625732, 2036.0914307, -2730.6760254, 2417.2941895

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 36

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0754006, upper bound: 1757.0433232
time: 0.70 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0755939, upper bound: 1757.0433211
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -363.4499512, 1473.1496582, -408.5358582, 1686.5399170, -2049.9897461, 1881.6853027
1: -445.7073975, 1644.7882080, -499.8553162, 1884.0002441, -2329.7075195, 2144.6428223
2: -510.4920959, 1672.4487305, -578.4812012, 1909.8503418, -2420.3425293, 2250.9299316
3: -724.4760132, 1825.3642578, -818.3936157, 2091.8496094, -2816.3254395, 2643.7570801
4: -858.0468750, 1701.8101807, -979.1275635, 1942.3623047, -2800.4086914, 2680.9370117

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0944415, upper bound: 1757.0923174
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0944415, upper bound: 1757.0923174
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -363.6856995, 1474.3394775, -413.1867065, 1705.2510986, -2068.9365234, 1887.5261230
1: -445.9946594, 1646.1282959, -505.5032043, 1904.8597412, -2350.8540039, 2151.6308594
2: -510.8197937, 1673.7861328, -585.0932007, 1931.1904297, -2442.0102539, 2258.8793945
3: -724.9641113, 1826.7500000, -827.5491943, 2114.9980469, -2839.9621582, 2654.2993164
4: -858.6025391, 1703.1427002, -990.2218628, 1963.9223633, -2822.5246582, 2693.3642578

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0944415, upper bound: 1757.0955368
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1012209, upper bound: 1757.0955368
time: 0.68 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -333.0895081, 1351.9078369, -432.4667358, 1780.7087402, -2113.7980957, 1784.3742676
1: -408.4187317, 1509.3395996, -528.4594727, 1988.2106934, -2396.6291504, 2037.7987061
2: -467.5669556, 1534.5799561, -613.2951050, 2016.0672607, -2483.6340332, 2147.8750000
3: -664.0418091, 1673.9576416, -866.1811523, 2214.1142578, -2878.1557617, 2540.1386719
4: -785.8018188, 1561.4708252, -1040.6778564, 2052.2551270, -2838.0568848, 2602.1477051

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0860896, upper bound: 1757.0528544
time: 0.66 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0811903, upper bound: 1757.0525850
time: 0.71 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -340.3004150, 1380.2144775, -439.7156677, 1813.8632812, -2154.1635742, 1819.9300537
1: -417.1874695, 1540.8438721, -537.8480835, 2025.2777100, -2442.4650879, 2078.6918945
2: -477.9605408, 1566.7092285, -623.7086182, 2053.1723633, -2531.1328125, 2190.4179688
3: -678.3616943, 1709.8724365, -881.5455933, 2253.1630859, -2931.5246582, 2591.4179688
4: -803.6917114, 1594.1402588, -1057.9991455, 2089.1699219, -2892.8615723, 2652.1386719

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0999657, upper bound: 1757.0943408
time: 0.71 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0999657, upper bound: 1757.0942408
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -363.3181152, 1472.5150146, -430.9279175, 1772.0571289, -2135.3752441, 1903.4428711
1: -445.6163940, 1644.1693115, -526.6234741, 1978.5738525, -2424.1901855, 2170.7927246
2: -509.9727173, 1671.8199463, -611.0308838, 2006.5072021, -2516.4799805, 2282.8508301
3: -724.0907593, 1823.6536865, -862.7700806, 2203.4648438, -2927.5556641, 2686.4238281
4: -856.4633179, 1701.2352295, -1036.2678223, 2042.7429199, -2899.2060547, 2737.5024414

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 31

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 29

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0827961, upper bound: 1757.0526689
time: 0.77 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -370.5737305, 1501.2280273, -438.3927612, 1805.9587402, -2176.5319824, 1939.6208496
1: -454.4270325, 1676.1418457, -536.2717896, 2016.4846191, -2470.9113770, 2212.4130859
2: -520.3884888, 1704.3659668, -621.7326660, 2044.5371094, -2564.9255371, 2326.0983887
3: -738.5195923, 1860.0334473, -878.5268555, 2243.4262695, -2981.9458008, 2738.5600586
4: -874.4147339, 1734.3714600, -1054.0551758, 2080.5742188, -2954.9882812, 2788.4265137

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1017159, upper bound: 1757.0945392
time: 0.80 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1017159, upper bound: 1757.0945392
time: 0.93 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -413.9170227, 1709.2200928, -307.7661438, 1251.7006836, -1665.6176758, 2016.9862061
1: -506.4060364, 1909.3297119, -377.5460815, 1398.2214355, -1904.6271973, 2286.8757324
2: -586.0026245, 1935.4637451, -430.7722778, 1420.9362793, -2006.9389648, 2366.2353516
3: -829.1879272, 2119.7893066, -613.1173096, 1546.5338135, -2375.7211914, 2732.9067383
4: -992.0071411, 1968.1402588, -722.2210693, 1444.7147217, -2436.7214355, 2690.3613281

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0917462, upper bound: 1757.0944555
time: 0.75 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0922356, upper bound: 1757.0925945
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0922356, upper bound: 1757.0936266
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -413.9170227, 1709.2200928, -347.9011536, 1409.8518066, -1823.7687988, 2057.1208496
1: -506.4060364, 1909.3297119, -426.4547119, 1574.0439453, -2080.4499512, 2335.7844238
2: -586.0026245, 1935.4637451, -488.4465027, 1600.3955078, -2186.3979492, 2423.9096680
3: -829.1879272, 2119.7893066, -693.1854248, 1746.0533447, -2575.2407227, 2812.9746094
4: -992.0071411, 1968.1402588, -820.9440918, 1628.1809082, -2620.1875000, 2789.0842285

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0917462, upper bound: 1757.0944555
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0922356, upper bound: 1757.0925945
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0922356, upper bound: 1757.0936266
time: 0.73 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -418.5886841, 1728.0035400, -308.8413696, 1256.6033936, -1675.1921387, 2036.8448486
1: -512.0783081, 1930.2691650, -378.8524780, 1403.6960449, -1915.7741699, 2309.1210938
2: -592.6408691, 1956.8780518, -432.3103638, 1426.4732666, -2019.1141357, 2389.1879883
3: -838.3815918, 2143.0317383, -615.3218384, 1552.5426025, -2390.9238281, 2758.3532715
4: -1003.1513672, 1989.7790527, -724.9045410, 1450.2597656, -2453.4111328, 2714.6835938

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0956148, upper bound: 1757.0982412
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0955990, upper bound: 1757.1012410
time: 1.00 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -418.5886841, 1728.0035400, -348.1538696, 1411.1296387, -1829.7182617, 2076.1574707
1: -512.0783081, 1930.2691650, -426.7574158, 1575.4812012, -2087.5595703, 2357.0263672
2: -592.6408691, 1956.8780518, -488.7874451, 1601.8282471, -2194.4692383, 2445.6652832
3: -838.3815918, 2143.0317383, -693.7021484, 1747.5516357, -2585.9331055, 2836.7336426
4: -1003.1513672, 1989.7790527, -821.5203857, 1629.6134033, -2632.7644043, 2811.2993164

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0956148, upper bound: 1757.0982412
time: 0.73 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0955990, upper bound: 1757.1012410
time: 0.79 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -413.9170227, 1709.2200928, -416.6647949, 1720.5021973, -2134.4191895, 2125.8845215
1: -506.4060364, 1909.3297119, -509.6967773, 1921.9409180, -2428.3469238, 2419.0263672
2: -586.0026245, 1935.4637451, -589.9318848, 1948.2512207, -2534.2536621, 2525.3952637
3: -829.1879272, 2119.7893066, -834.5861816, 2133.8591309, -2963.0468750, 2954.3754883
4: -992.0071411, 1968.1402588, -998.7283325, 1981.1147461, -2973.1218262, 2966.8686523

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0917462, upper bound: 1757.0936850
time: 0.79 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0922211, upper bound: 1757.0921865
time: 0.78 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0922211, upper bound: 1757.0921865
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -413.9170227, 1709.2200928, -442.0362549, 1821.9112549, -2235.8283691, 2151.2563477
1: -506.4060364, 1909.3297119, -540.7137451, 2034.3081055, -2540.7138672, 2450.0434570
2: -586.0026245, 1935.4637451, -626.7739868, 2062.5097656, -2648.5117188, 2562.2370605
3: -829.1879272, 2119.7893066, -885.9022217, 2262.8747559, -3092.0617676, 3005.6914062
4: -992.0071411, 1968.1402588, -1062.6588135, 2098.6401367, -3090.6469727, 3030.7990723

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0917462, upper bound: 1757.0936850
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0922211, upper bound: 1757.0921865
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0922211, upper bound: 1757.0921865
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -418.5886841, 1728.0035400, -417.2853699, 1722.8443604, -2141.4331055, 2145.2888184
1: -512.0783081, 1930.2691650, -510.4707031, 1924.5747070, -2436.6530762, 2440.7397461
2: -592.6408691, 1956.8780518, -590.7763672, 1951.0024414, -2543.6433105, 2547.6542969
3: -838.3815918, 2143.0317383, -835.7814941, 2136.6579590, -2975.0395508, 2978.8129883
4: -1003.1513672, 1989.7790527, -1000.0374146, 1983.8791504, -2987.0305176, 2989.8164062

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0954948, upper bound: 1757.0943454
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0954730, upper bound: 1757.0944453
time: 0.98 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -418.5886841, 1728.0035400, -442.4679565, 1823.6433105, -2242.2319336, 2170.4714355
1: -512.0783081, 1930.2691650, -541.2393188, 2036.2568359, -2548.3347168, 2471.5083008
2: -592.6408691, 1956.8780518, -627.3767700, 2064.5207520, -2657.1616211, 2584.2543945
3: -838.3815918, 2143.0317383, -886.7295532, 2264.9287109, -3103.3103027, 3029.7607422
4: -1003.1513672, 1989.7790527, -1063.6060791, 2100.6066895, -3103.7580566, 3053.3852539

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 36

### Candidate
type: A, layer: 1, pos: 23

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 30

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 31

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 20

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 15

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 41

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 3

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0954948, upper bound: 1757.0943454
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0954730, upper bound: 1757.0944453
time: 0.81 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 5.77 seconds
NS_A1_B1_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0940500, upper bound: 1757.1106456
NS_A1_B1_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1174104, upper bound: 1757.1172664
NS_A1_B1_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0940500, upper bound: 1757.1106456
NS_A1_B1_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1174104, upper bound: 1757.1175706
NS_A1_B1_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1045515, upper bound: 1757.1134242
NS_A1_B1_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1045515, upper bound: 1757.1164502
NS_A1_B1_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1045515, upper bound: 1757.1134242
NS_A1_B1_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1131634, upper bound: 1757.1165374
NS_A1_B1_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0568420, upper bound: 1757.0851416
NS_A1_B1_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0579947, upper bound: 1757.0845757
NS_A1_B1_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0568420, upper bound: 1757.0851416
NS_A1_B1_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0579947, upper bound: 1757.0845757
NS_A1_B1_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0624251, upper bound: 1757.0886251
NS_A1_B1_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1047296, upper bound: 1757.1017128
NS_A1_B1_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0624251, upper bound: 1757.0886251
NS_A1_B1_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1047296, upper bound: 1757.1017128
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1081712, upper bound: 1757.1102394
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1081712, upper bound: 1757.1102394
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1081712, upper bound: 1757.1102394
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1081712, upper bound: 1757.1102394
NS_A1_B1_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
NS_A1_B1_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
NS_A1_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
NS_A1_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1028412, upper bound: 1757.1098000
NS_A1_B1_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1074923, upper bound: 1757.1026508
NS_A1_B1_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1074923, upper bound: 1757.1026508
NS_A1_B1_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1074923, upper bound: 1757.1026508
NS_A1_B1_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1074923, upper bound: 1757.1026508
NS_A1_B1_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
NS_A1_B1_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
NS_A1_B1_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
NS_A1_B1_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1024392, upper bound: 1757.1024392
NS_A1_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0754006, upper bound: 1757.0416133
NS_A1_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0755938, upper bound: 1757.0416238
NS_A1_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0754006, upper bound: 1757.0433232
NS_A1_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0755939, upper bound: 1757.0433211
NS_A1_B2_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0944415, upper bound: 1757.0923174
NS_A1_B2_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0944415, upper bound: 1757.0923174
NS_A1_B2_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0944415, upper bound: 1757.0955368
NS_A1_B2_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1012209, upper bound: 1757.0955368
NS_A1_B2_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0860896, upper bound: 1757.0528544
NS_A1_B2_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0811903, upper bound: 1757.0525850
NS_A1_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0999657, upper bound: 1757.0943408
NS_A1_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0999657, upper bound: 1757.0942408
NS_A1_B2_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1017159, upper bound: 1757.0945392
NS_A1_B2_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.1017159, upper bound: 1757.0945392
NS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0922356, upper bound: 1757.0925945
NS_A2_B2_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0922356, upper bound: 1757.0936266
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0922356, upper bound: 1757.0925945
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0922356, upper bound: 1757.0936266
NS_A2_B2_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0956148, upper bound: 1757.0982412
NS_A2_B2_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0955990, upper bound: 1757.1012410
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0956148, upper bound: 1757.0982412
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0955990, upper bound: 1757.1012410
NS_A2_B2_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0922211, upper bound: 1757.0921865
NS_A2_B2_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0922211, upper bound: 1757.0921865
NS_A2_B2_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0922211, upper bound: 1757.0921865
NS_A2_B2_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0922211, upper bound: 1757.0921865
NS_A2_B2_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0954948, upper bound: 1757.0943454
NS_A2_B2_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0954730, upper bound: 1757.0944453
NS_A2_B2_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0954948, upper bound: 1757.0943454
NS_A2_B2_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 5.77
Output dim: 0, lower bound: -1757.0954730, upper bound: 1757.0944453

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -263.0589600, 1076.6291504, -264.1922913, 1077.7495117, -1340.8083496, 1340.8214111
1: -322.2584534, 1202.1828613, -323.8448792, 1203.6641846, -1525.9226074, 1526.0277100
2: -368.3470459, 1221.4058838, -369.3798828, 1222.9600830, -1591.3071289, 1590.7857666
3: -523.5605469, 1328.4720459, -525.4531860, 1329.5482178, -1853.1087646, 1853.9250488
4: -617.3064575, 1241.0792236, -618.2068481, 1242.9171143, -1860.2236328, 1859.2861328

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1143182, upper bound: 1757.1137630
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1146860, upper bound: 1757.1150844
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -271.2267456, 1106.1020508, -271.4596863, 1106.8758545, -1378.1025391, 1377.5616455
1: -332.5411377, 1235.3898926, -332.8246460, 1236.2692871, -1568.8104248, 1568.2144775
2: -379.0629578, 1255.1336670, -379.3858337, 1256.0178223, -1635.0806885, 1634.5195312
3: -539.5594482, 1364.3531494, -539.9813843, 1365.3222656, -1904.8817139, 1904.3344727
4: -634.4192505, 1275.4968262, -634.9245605, 1276.4086914, -1910.8278809, 1910.4213867

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1112184, upper bound: 1757.1163623
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1109358, upper bound: 1757.1109358
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -263.0589600, 1076.6291504, -302.6303101, 1231.8994141, -1494.9583740, 1379.2595215
1: -322.2584534, 1202.1828613, -371.2094421, 1376.0196533, -1698.2780762, 1573.3923340
2: -368.3470459, 1221.4058838, -423.7509460, 1398.4240723, -1766.7711182, 1645.1568604
3: -523.5605469, 1328.4720459, -602.9878540, 1522.2205811, -2045.7808838, 1931.4598389
4: -617.3064575, 1241.0792236, -710.6203003, 1421.8841553, -2039.1906738, 1951.6994629

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0743942, upper bound: 1757.1019321
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0940500, upper bound: 1757.1106456
time: 0.78 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -271.2267456, 1106.1020508, -307.8125000, 1252.2995605, -1523.5263672, 1413.9144287
1: -332.5411377, 1235.3898926, -377.5932617, 1398.8896484, -1731.4306641, 1612.9831543
2: -379.0629578, 1255.1336670, -430.8732300, 1421.6036377, -1800.6665039, 1686.0068359
3: -539.5594482, 1364.3531494, -613.2637939, 1547.2501221, -2086.8095703, 1977.6169434
4: -634.4192505, 1275.4968262, -722.4856567, 1445.3218994, -2079.7409668, 1997.9822998

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1175380, upper bound: 1757.1174898
time: 0.85 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1175380, upper bound: 1757.1175706
time: 0.89 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -299.7027283, 1216.7593994, -264.9118958, 1080.2097168, -1379.9122314, 1481.6712646
1: -367.3187561, 1358.5539551, -324.8663940, 1206.5163574, -1573.8350830, 1683.4202881
2: -419.8541870, 1380.6575928, -370.1453857, 1225.6833496, -1645.5374756, 1750.8028564
3: -596.4208984, 1504.0637207, -527.0153198, 1332.0993652, -1928.5202637, 2031.0789795
4: -704.4074097, 1403.8482666, -619.2089233, 1245.4814453, -1949.8885498, 2023.0571289

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1042990, upper bound: 1757.1137341
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1041824, upper bound: 1757.1098818
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -324.3232117, 1315.8641357, -263.9649658, 1074.2398682, -1398.5629883, 1579.8286133
1: -397.3739929, 1469.2979736, -323.6406250, 1199.9460449, -1597.3200684, 1792.9383545
2: -454.1462402, 1493.4281006, -368.6650391, 1219.1561279, -1673.3023682, 1862.0931396
3: -644.8476562, 1625.8554688, -524.6296387, 1324.7900391, -1969.6376953, 2150.4851074
4: -760.8547974, 1518.2220459, -616.1744995, 1239.0850830, -1999.9396973, 2134.3964844

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1128979, upper bound: 1757.1152442
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1138418, upper bound: 1757.1170251
time: 0.77 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -299.7027283, 1216.7593994, -301.2296753, 1225.6713867, -1525.3737793, 1517.9890137
1: -367.3187561, 1358.5539551, -369.5244141, 1369.2202148, -1736.5389404, 1728.0783691
2: -419.8541870, 1380.6575928, -421.4910889, 1391.3468018, -1811.2009277, 1802.1486816
3: -596.4208984, 1504.0637207, -600.0631714, 1513.6772461, -2110.0981445, 2104.1269531
4: -704.4074097, 1403.8482666, -706.4505615, 1414.3660889, -2118.7734375, 2110.2980957

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1043303, upper bound: 1757.1126870
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1042375, upper bound: 1757.1101423
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -324.3232117, 1315.8641357, -300.9036865, 1222.4104004, -1546.7333984, 1616.7677002
1: -397.3739929, 1469.2979736, -369.1870422, 1365.5828857, -1762.9569092, 1838.4849854
2: -454.1462402, 1493.4281006, -421.0191040, 1387.8005371, -1841.9467773, 1914.4472656
3: -644.8476562, 1625.8554688, -599.2694092, 1510.3040771, -2155.1518555, 2225.1250000
4: -760.8547974, 1518.2220459, -705.4302979, 1411.2031250, -2172.0578613, 2223.6518555

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1109529, upper bound: 1757.1156911
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1106944, upper bound: 1757.1111454
time: 1.14 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -294.8867798, 1187.6541748, -295.5489197, 1200.1967773, -1495.0833740, 1483.2031250
1: -361.0476379, 1326.6569824, -362.2709351, 1340.1024170, -1701.1499023, 1688.9279785
2: -412.6512146, 1347.7526855, -413.7465210, 1361.9335938, -1774.5847168, 1761.4991455
3: -585.2242432, 1472.3449707, -588.0549316, 1482.9421387, -2068.1665039, 2060.3999023
4: -692.6013184, 1371.7473145, -693.6113281, 1384.8499756, -2077.4506836, 2065.3586426

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0551908, upper bound: 1757.0567716
time: 0.95 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0551908, upper bound: 1757.0886616
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -318.8774719, 1285.4371338, -295.1134033, 1197.0791016, -1515.9565430, 1580.5505371
1: -390.4550476, 1435.8284912, -361.6855164, 1336.7210693, -1727.1760254, 1797.5139160
2: -446.3908386, 1458.9932861, -413.0191040, 1358.5660400, -1804.9569092, 1872.0124512
3: -632.8867188, 1592.4105225, -586.8380737, 1479.1586914, -2112.0444336, 2179.2485352
4: -748.4069214, 1484.5078125, -692.0132446, 1381.6517334, -2130.0581055, 2176.5205078

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0570390, upper bound: 1757.0570390
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0570390, upper bound: 1757.0888033
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -294.8867798, 1187.6541748, -332.2426758, 1347.1708984, -1642.0576172, 1519.8968506
1: -361.0476379, 1326.6569824, -407.3606262, 1504.1998291, -1865.2473145, 1734.0175781
2: -412.6512146, 1347.7526855, -466.1146545, 1529.2471924, -1941.8984375, 1813.8673096
3: -585.2242432, 1472.3449707, -661.9687500, 1667.1907959, -2252.4150391, 2134.3137207
4: -692.6013184, 1371.7473145, -782.7709351, 1555.6198730, -2248.2211914, 2154.5183105

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0567521, upper bound: 1757.0842475
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0568420, upper bound: 1757.0851416
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -318.8774719, 1285.4371338, -332.7547302, 1347.8503418, -1666.7277832, 1618.1918945
1: -390.4550476, 1435.8284912, -408.0347900, 1504.9802246, -1895.4350586, 1843.8632812
2: -446.3908386, 1458.9932861, -466.8380432, 1530.1700439, -1976.5606689, 1925.8312988
3: -632.8867188, 1592.4105225, -662.9020386, 1668.4959717, -2301.3818359, 2255.3125000
4: -748.4069214, 1484.5078125, -783.8632202, 1556.7868652, -2305.1936035, 2268.3710938

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0575972, upper bound: 1757.0835991
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0579947, upper bound: 1757.0845757
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -264.3359070, 1075.2929688, -282.8289490, 1149.9168701, -1414.2525635, 1358.1217041
1: -323.9374390, 1200.6091309, -346.6604614, 1283.8522949, -1607.7895508, 1547.2695312
2: -371.0682983, 1220.4807129, -396.0849304, 1305.0549316, -1676.1231689, 1616.5655518
3: -526.2934570, 1329.1878662, -562.5112915, 1419.9693604, -1946.2628174, 1891.6989746
4: -622.3557739, 1241.5869141, -663.2594604, 1327.0438232, -1949.3991699, 1904.8461914

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1757.0687110, upper bound: 1757.0687110
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0687110, upper bound: 1757.0983265
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -303.8256531, 1232.4853516, -306.3143921, 1242.9261475, -1546.7517090, 1538.7998047
1: -372.3684692, 1375.9885254, -375.3900146, 1387.6604004, -1760.0288086, 1751.3785400
2: -425.6007690, 1398.7282715, -429.1296692, 1410.5220947, -1836.1226807, 1827.8579102
3: -604.4763794, 1523.5635986, -609.4556274, 1536.5571289, -2141.0332031, 2133.0187988
4: -713.9244995, 1422.3046875, -719.9587402, 1434.3107910, -2148.2353516, 2142.2634277

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1068541, upper bound: 1757.1076148
time: 0.74 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1076147, upper bound: 1757.1076148
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -264.3359070, 1075.2929688, -319.5080872, 1296.4984131, -1560.8339844, 1394.8010254
1: -323.9374390, 1200.6091309, -391.7665405, 1447.5288086, -1771.4660645, 1592.3757324
2: -371.0682983, 1220.4807129, -448.5035400, 1471.9094238, -1842.9776611, 1668.9840088
3: -526.2934570, 1329.1878662, -636.5300293, 1603.7252197, -2130.0185547, 1965.7178955
4: -622.3557739, 1241.5869141, -752.7537842, 1497.3190918, -2119.6748047, 1994.3406982

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 42

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0620182, upper bound: 1757.0885465
time: 0.62 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.0624251, upper bound: 1757.0886251
time: 0.70 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -303.8256531, 1232.4853516, -343.5640564, 1391.6252441, -1695.4508057, 1576.0493164
1: -372.3684692, 1375.9885254, -421.1633911, 1553.6236572, -1925.9921875, 1797.1518555
2: -425.6007690, 1398.7282715, -482.2944031, 1579.7772217, -2005.3779297, 1881.0227051
3: -604.4763794, 1523.5635986, -684.4097290, 1723.2164307, -2327.6926270, 2207.9729004
4: -713.9244995, 1422.3046875, -810.3569336, 1607.0511475, -2320.9753418, 2232.6616211

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 41

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 23

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1041781, upper bound: 1757.1013137
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1047296, upper bound: 1757.1017128
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -301.1167297, 1226.5225830, -302.4146423, 1231.6340332, -1532.7507324, 1528.9371338
1: -369.4195862, 1370.0664062, -371.0033875, 1375.7783203, -1745.1978760, 1741.0697021
2: -421.6261597, 1392.2893066, -423.4243774, 1398.1044922, -1819.7304688, 1815.7135010
3: -600.2796021, 1515.8808594, -602.8259277, 1522.2109375, -2122.4904785, 2118.7067871
4: -707.2943115, 1416.0643311, -710.2945557, 1421.9650879, -2129.2592773, 2126.3588867

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1101365, upper bound: 1757.1156379
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1098996, upper bound: 1757.1111615
time: 0.90 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -341.1630249, 1383.8427734, -302.4146423, 1231.6340332, -1572.7969971, 1686.2572021
1: -418.2403870, 1544.8682861, -371.0033875, 1375.7783203, -1794.0186768, 1915.8717041
2: -479.1471252, 1570.8157959, -423.4243774, 1398.1044922, -1877.2515869, 1994.2399902
3: -680.0970459, 1714.2969971, -602.8259277, 1522.2109375, -2202.3081055, 2317.1230469
4: -805.6978760, 1598.2816162, -710.2945557, 1421.9650879, -2227.6628418, 2308.5759277

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1101365, upper bound: 1757.1156379
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1098996, upper bound: 1757.1111615
time: 0.71 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -301.1167297, 1226.5225830, -331.6198425, 1348.4567871, -1649.5734863, 1558.1420898
1: -369.4195862, 1370.0664062, -406.9620056, 1506.3532715, -1875.7728271, 1777.0283203
2: -421.6261597, 1392.2893066, -464.4823608, 1531.0493164, -1952.6754150, 1856.7714844
3: -600.2796021, 1515.8808594, -660.9517822, 1667.2908936, -2267.5705566, 2176.8325195
4: -707.2943115, 1416.0643311, -778.8173828, 1557.4371338, -2264.7314453, 2194.8818359

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1074354, upper bound: 1757.1094296
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1073900, upper bound: 1757.1085044
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -341.1630249, 1383.8427734, -331.6198425, 1348.4567871, -1689.6197510, 1715.4621582
1: -418.2403870, 1544.8682861, -406.9620056, 1506.3532715, -1924.5936279, 1951.8303223
2: -479.1471252, 1570.8157959, -464.4823608, 1531.0493164, -2010.1964111, 2035.2980957
3: -680.0970459, 1714.2969971, -660.9517822, 1667.2908936, -2347.3879395, 2375.2487793
4: -805.6978760, 1598.2816162, -778.8173828, 1557.4371338, -2363.1350098, 2377.0988770

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1074354, upper bound: 1757.1094296
time: 0.72 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1073900, upper bound: 1757.1085044
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -330.7549133, 1344.9018555, -302.4146423, 1231.6340332, -1562.3889160, 1647.3165283
1: -405.9093018, 1502.3511963, -371.0033875, 1375.7783203, -1781.6875000, 1873.3546143
2: -463.2861938, 1527.0256348, -423.4243774, 1398.1044922, -1861.3905029, 1950.4497070
3: -659.2329712, 1662.9183350, -602.8259277, 1522.2109375, -2181.4438477, 2265.7441406
4: -776.7978516, 1553.3138428, -710.2945557, 1421.9650879, -2198.7629395, 2263.6081543

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025776, upper bound: 1757.1129333
time: 1.14 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024016, upper bound: 1757.1101138
time: 1.04 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -371.4362183, 1504.7821045, -302.4146423, 1231.6340332, -1603.0700684, 1807.1966553
1: -455.4780273, 1680.0803223, -371.0033875, 1375.7783203, -1831.2559814, 2051.0837402
2: -521.5752563, 1708.3962402, -423.4243774, 1398.1044922, -1919.6796875, 2131.8205566
3: -740.2426147, 1864.3790283, -602.8259277, 1522.2109375, -2262.4536133, 2467.2045898
4: -876.4083862, 1738.4490967, -710.2945557, 1421.9650879, -2298.3735352, 2448.7436523

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1025776, upper bound: 1757.1129333
time: 0.70 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1024016, upper bound: 1757.1101138
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -330.7549133, 1344.9018555, -331.6198425, 1348.4567871, -1679.2116699, 1676.5213623
1: -405.9093018, 1502.3511963, -406.9620056, 1506.3532715, -1912.2625732, 1909.3132324
2: -463.2861938, 1527.0256348, -464.4823608, 1531.0493164, -1994.3354492, 1991.5076904
3: -659.2329712, 1662.9183350, -660.9517822, 1667.2908936, -2326.5236816, 2323.8701172
4: -776.7978516, 1553.3138428, -778.8173828, 1557.4371338, -2334.2348633, 2332.1311035

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1023340, upper bound: 1757.1089958
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022860, upper bound: 1757.1081486
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -371.4362183, 1504.7821045, -331.6198425, 1348.4567871, -1719.8928223, 1836.4016113
1: -455.4780273, 1680.0803223, -406.9620056, 1506.3532715, -1961.8308105, 2087.0419922
2: -521.5752563, 1708.3962402, -464.4823608, 1531.0493164, -2052.6242676, 2172.8786621
3: -740.2426147, 1864.3790283, -660.9517822, 1667.2908936, -2407.5334473, 2525.3308105
4: -876.4083862, 1738.4490967, -778.8173828, 1557.4371338, -2433.8454590, 2517.2666016

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1023340, upper bound: 1757.1089958
time: 0.73 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1022860, upper bound: 1757.1081486
time: 0.72 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -301.1167297, 1226.5225830, -342.5917358, 1389.5111084, -1690.6278076, 1569.1142578
1: -369.4195862, 1370.0664062, -419.9836426, 1551.1816406, -1920.6011963, 1790.0499268
2: -421.6261597, 1392.2893066, -481.1403809, 1577.2651367, -1998.8911133, 1873.4295654
3: -600.2796021, 1515.8808594, -682.9180298, 1721.3345947, -2321.6140137, 2198.7988281
4: -707.2943115, 1416.0643311, -809.0411377, 1604.8242188, -2312.1181641, 2225.1052246

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1106667, upper bound: 1757.1102664
time: 0.79 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1106667, upper bound: 1757.1106667
time: 0.68 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -341.1630249, 1383.8427734, -342.5917358, 1389.5111084, -1730.6740723, 1726.4343262
1: -418.2403870, 1544.8682861, -419.9836426, 1551.1816406, -1969.4219971, 1964.8519287
2: -479.1471252, 1570.8157959, -481.1403809, 1577.2651367, -2056.4123535, 2051.9560547
3: -680.0970459, 1714.2969971, -682.9180298, 1721.3345947, -2401.4316406, 2397.2150879
4: -805.6978760, 1598.2816162, -809.0411377, 1604.8242188, -2410.5219727, 2407.3225098

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1106667, upper bound: 1757.1102664
time: 0.77 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1106667, upper bound: 1757.1106667
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -301.1167297, 1226.5225830, -372.8626404, 1510.5230713, -1811.6397705, 1599.3852539
1: -369.4195862, 1370.0664062, -457.2156067, 1686.4685059, -2055.8879395, 1827.2816162
2: -421.6261597, 1392.2893066, -523.5686646, 1714.9011230, -2136.5273438, 1915.8579102
3: -600.2796021, 1515.8808594, -743.0819702, 1871.5242920, -2471.8037109, 2258.9628906
4: -707.2943115, 1416.0643311, -879.7874146, 1745.0615234, -2452.3559570, 2295.8518066

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063053, upper bound: 1757.1007562
time: 0.59 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1067779, upper bound: 1757.1021764
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -341.1630249, 1383.8427734, -372.8626404, 1510.5230713, -1851.6860352, 1756.7052002
1: -418.2403870, 1544.8682861, -457.2156067, 1686.4685059, -2104.7087402, 2002.0836182
2: -479.1471252, 1570.8157959, -523.5686646, 1714.9011230, -2194.0483398, 2094.3845215
3: -680.0970459, 1714.2969971, -743.0819702, 1871.5242920, -2551.6213379, 2457.3789062
4: -805.6978760, 1598.2816162, -879.7874146, 1745.0615234, -2550.7592773, 2478.0690918

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 15

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1063053, upper bound: 1757.1007562
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1067779, upper bound: 1757.1021764
time: 0.76 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -330.7549133, 1344.9018555, -342.5917358, 1389.5111084, -1720.2659912, 1687.4935303
1: -405.9093018, 1502.3511963, -419.9836426, 1551.1816406, -1957.0906982, 1922.3348389
2: -463.2861938, 1527.0256348, -481.1403809, 1577.2651367, -2040.5512695, 2008.1657715
3: -659.2329712, 1662.9183350, -682.9180298, 1721.3345947, -2380.5676270, 2345.8364258
4: -776.7978516, 1553.3138428, -809.0411377, 1604.8242188, -2381.6218262, 2362.3544922

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1020573, upper bound: 1757.1063689
time: 0.78 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1021764, upper bound: 1757.1067779
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -371.4362183, 1504.7821045, -342.5917358, 1389.5111084, -1760.9472656, 1847.3737793
1: -455.4780273, 1680.0803223, -419.9836426, 1551.1816406, -2006.6591797, 2100.0629883
2: -521.5752563, 1708.3962402, -481.1403809, 1577.2651367, -2098.8403320, 2189.5363770
3: -740.2426147, 1864.3790283, -682.9180298, 1721.3345947, -2461.5771484, 2547.2968750
4: -876.4083862, 1738.4490967, -809.0411377, 1604.8242188, -2481.2321777, 2547.4902344

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 32

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 36

### Candidate
type: B, layer: 1, pos: 29

### Candidate
type: B, layer: 1, pos: 30

### Candidate
type: B, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 15

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 31

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1020573, upper bound: 1757.1063689
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1757.1021764, upper bound: 1757.1067779
time: 0.80 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.52 + 416.70 = 420.22 seconds
