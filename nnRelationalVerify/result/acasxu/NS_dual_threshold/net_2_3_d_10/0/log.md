## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 0)
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
execution time: IAR + RelationalAnalysis = 1.63 + 1.86 = 3.49 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -1379.3191890, upper bound: 1379.3191890

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 37

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3182631, upper bound: 1379.3171600
time: 0.63 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3191890, upper bound: 1379.3191890
time: 0.87 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.64 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 0, lower bound: -1379.3182631, upper bound: 1379.3171600
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.64
Output dim: 0, lower bound: -1379.3191890, upper bound: 1379.3191890

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -381.7454224, 1149.8890381, -382.7834778, 1154.7761230, -1536.5213623, 1532.6724854
1: -295.1368713, 764.2340698, -296.1175537, 767.3020630, -1062.4387207, 1060.3515625
2: -319.2566223, 726.4481812, -320.3592529, 729.6099854, -1048.8665771, 1046.8073730
3: -320.8620605, 930.9282227, -322.0329285, 934.6010132, -1255.4628906, 1252.9610596
4: -465.6395264, 787.1626587, -467.4971313, 790.5286255, -1256.1679688, 1254.6596680

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3171600, upper bound: 1379.3171600
time: 0.79 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3171600, upper bound: 1379.3171600
time: 0.78 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -378.3921814, 1141.9267578, -383.7346802, 1157.7136230, -1536.1058350, 1525.6613770
1: -292.6275330, 758.4955444, -296.8671875, 769.2732544, -1061.9007568, 1055.3627930
2: -316.5532837, 721.2775879, -321.1560059, 731.4843140, -1048.0374756, 1042.4335938
3: -318.3967285, 923.9514771, -322.8305969, 936.9838257, -1255.3806152, 1246.7821045
4: -462.0427246, 781.3889771, -468.6798096, 792.5546875, -1254.5971680, 1250.0688477

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3171600, upper bound: 1379.3182631
time: 1.06 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3171600, upper bound: 1379.3191890
time: 0.76 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.45 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -1379.3171600, upper bound: 1379.3171600
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -1379.3171600, upper bound: 1379.3171600
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -1379.3171600, upper bound: 1379.3182631
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.45
Output dim: 0, lower bound: -1379.3171600, upper bound: 1379.3191890

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -381.7454224, 1149.8890381, -381.7454224, 1149.8890381, -1531.6341553, 1531.6341553
1: -295.1368713, 764.2340698, -295.1368713, 764.2340698, -1059.3709717, 1059.3709717
2: -319.2566223, 726.4481812, -319.2566223, 726.4481812, -1045.7048340, 1045.7048340
3: -320.8620605, 930.9282227, -320.8620605, 930.9282227, -1251.7900391, 1251.7900391
4: -465.6395264, 787.1626587, -465.6395264, 787.1626587, -1252.8020020, 1252.8018799

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134371, upper bound: 1379.3168282
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
time: 0.84 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -381.7454224, 1149.8890381, -378.3921814, 1141.9267578, -1523.6718750, 1528.2812500
1: -295.1368713, 764.2340698, -292.6275330, 758.4955444, -1053.6324463, 1056.8615723
2: -319.2566223, 726.4481812, -316.5532837, 721.2775879, -1040.5341797, 1043.0014648
3: -320.8620605, 930.9282227, -318.3967285, 923.9514771, -1244.8133545, 1249.3249512
4: -465.6395264, 787.1626587, -462.0427246, 781.3889771, -1247.0285645, 1249.2049561

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3167587, upper bound: 1379.3134344
time: 0.78 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
time: 0.53 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -378.3921814, 1141.9267578, -381.7454224, 1149.8890381, -1528.2812500, 1523.6718750
1: -292.6275330, 758.4955444, -295.1368713, 764.2340698, -1056.8615723, 1053.6324463
2: -316.5532837, 721.2775879, -319.2566223, 726.4481812, -1043.0014648, 1040.5341797
3: -318.3967285, 923.9514771, -320.8620605, 930.9282227, -1249.3249512, 1244.8133545
4: -462.0427246, 781.3889771, -465.6395264, 787.1626587, -1249.2049561, 1247.0285645

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134344, upper bound: 1379.3167586
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
time: 0.72 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -378.3921814, 1141.9267578, -378.3921814, 1141.9267578, -1520.3189697, 1520.3189697
1: -292.6275330, 758.4955444, -292.6275330, 758.4955444, -1051.1230469, 1051.1230469
2: -316.5532837, 721.2775879, -316.5532837, 721.2775879, -1037.8308105, 1037.8308105
3: -318.3967285, 923.9514771, -318.3967285, 923.9514771, -1242.3481445, 1242.3481445
4: -462.0427246, 781.3889771, -462.0427246, 781.3889771, -1243.4315186, 1243.4315186

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

## Relational analysis of NS_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134344, upper bound: 1379.3172248
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
time: 0.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.71 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.71
Output dim: 0, lower bound: -1379.3134371, upper bound: 1379.3168282
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.71
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.71
Output dim: 0, lower bound: -1379.3167587, upper bound: 1379.3134344
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.71
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.71
Output dim: 0, lower bound: -1379.3134344, upper bound: 1379.3167586
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.71
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.71
Output dim: 0, lower bound: -1379.3134344, upper bound: 1379.3172248
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.71
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -378.1729126, 1139.0440674, -381.7454224, 1149.8890381, -1528.0615234, 1520.7893066
1: -292.3612061, 757.0189819, -295.1368713, 764.2340698, -1056.5952148, 1052.1557617
2: -316.2933960, 719.6064453, -319.2566223, 726.4481812, -1042.7415771, 1038.8630371
3: -317.8445129, 922.1145020, -320.8620605, 930.9282227, -1248.7727051, 1242.9763184
4: -461.2918091, 779.7399292, -465.6395264, 787.1626587, -1248.4539795, 1245.3793945

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
time: 0.60 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -382.3175964, 1150.1569824, -379.7151489, 1143.6855469, -1526.0030518, 1529.8720703
1: -295.2808228, 764.3681030, -293.5686035, 760.1306152, -1055.4113770, 1057.9367676
2: -319.3530273, 726.3499146, -317.5846558, 722.5502930, -1041.9033203, 1043.9345703
3: -321.0665894, 931.2224121, -319.1447754, 925.9264526, -1246.9927979, 1250.3671875
4: -465.7024536, 787.0157471, -463.1661987, 782.9396362, -1248.6419678, 1250.1818848

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -381.7454224, 1149.8890381, -374.8723755, 1131.2322998, -1512.9774170, 1524.7614746
1: -295.1368713, 764.2340698, -289.8943787, 751.3854980, -1046.5223389, 1054.1282959
2: -319.2566223, 726.4481812, -313.6442871, 714.5298462, -1033.7863770, 1040.0925293
3: -320.8620605, 930.9282227, -315.4222412, 915.2698364, -1236.1317139, 1246.3504639
4: -465.6395264, 787.1626587, -457.7706299, 774.0721436, -1239.7114258, 1244.9329834

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3137845, upper bound: 1379.3134119
time: 0.58 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3137845, upper bound: 1379.3134119
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -379.7151489, 1143.6855469, -376.9593506, 1136.4855957, -1516.2006836, 1520.6447754
1: -293.5686035, 760.1306152, -291.3263855, 754.8334961, -1048.4019775, 1051.4570312
2: -317.5846558, 722.5502930, -315.1031189, 717.6030884, -1035.1877441, 1037.6530762
3: -319.1447754, 925.9264526, -317.0129089, 919.6062012, -1238.7507324, 1242.9392090
4: -463.1661987, 782.9396362, -459.8348999, 777.3976440, -1240.5637207, 1242.7741699

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3137845, upper bound: 1379.3134119
time: 0.70 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3137845, upper bound: 1379.3134119
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -374.8723755, 1131.2322998, -381.7454224, 1149.8890381, -1524.7614746, 1512.9774170
1: -289.8943787, 751.3854980, -295.1368713, 764.2340698, -1054.1284180, 1046.5223389
2: -313.6442871, 714.5298462, -319.2566223, 726.4481812, -1040.0925293, 1033.7863770
3: -315.4222412, 915.2698364, -320.8620605, 930.9282227, -1246.3504639, 1236.1317139
4: -457.7706299, 774.0721436, -465.6395264, 787.1626587, -1244.9329834, 1239.7113037

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
time: 0.64 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -376.9593506, 1136.4855957, -379.7151489, 1143.6855469, -1520.6447754, 1516.2006836
1: -291.3263855, 754.8334961, -293.5686035, 760.1306152, -1051.4570312, 1048.4019775
2: -315.1031189, 717.6030884, -317.5846558, 722.5502930, -1037.6530762, 1035.1877441
3: -317.0129089, 919.6062012, -319.1447754, 925.9264526, -1242.9392090, 1238.7507324
4: -459.8348999, 777.3976440, -463.1661987, 782.9396362, -1242.7741699, 1240.5637207

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
time: 0.65 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -374.8723755, 1131.2322998, -378.3921814, 1141.9267578, -1516.7990723, 1509.6245117
1: -289.8943787, 751.3854980, -292.6275330, 758.4955444, -1048.3898926, 1044.0130615
2: -313.6442871, 714.5298462, -316.5532837, 721.2775879, -1034.9218750, 1031.0828857
3: -315.4222412, 915.2698364, -318.3967285, 923.9514771, -1239.3737793, 1233.6665039
4: -457.7706299, 774.0721436, -462.0427246, 781.3889771, -1239.1596680, 1236.1145020

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3137808, upper bound: 1379.3137845
time: 1.08 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3137808, upper bound: 1379.3137845
time: 1.05 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -376.9593506, 1136.4855957, -376.2556458, 1135.4011230, -1512.3604736, 1512.7412109
1: -291.3263855, 754.8334961, -290.9815674, 754.1779785, -1045.5043945, 1045.8149414
2: -315.1031189, 717.6030884, -314.7941895, 717.1758423, -1032.2789307, 1032.3972168
3: -317.0129089, 919.6062012, -316.5928955, 918.6853638, -1235.6981201, 1236.1987305
4: -459.8348999, 777.3976440, -459.4367981, 776.9492188, -1236.7841797, 1236.8343506

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
time: 0.81 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.10 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3134119
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3137845, upper bound: 1379.3134119
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3137845, upper bound: 1379.3134119
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3137845, upper bound: 1379.3134119
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3137845, upper bound: 1379.3134119
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3137808, upper bound: 1379.3137845
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3137808, upper bound: 1379.3137845
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.10
Output dim: 0, lower bound: -1379.3134119, upper bound: 1379.3137845

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -378.1729126, 1139.0440674, -378.1729126, 1139.0440674, -1517.2166748, 1517.2166748
1: -292.3612061, 757.0189819, -292.3612061, 757.0189819, -1049.3800049, 1049.3800049
2: -316.2933960, 719.6064453, -316.2933960, 719.6064453, -1035.8999023, 1035.8999023
3: -317.8445129, 922.1145020, -317.8445129, 922.1145020, -1239.9589844, 1239.9589844
4: -461.2918091, 779.7399292, -461.2918091, 779.7399292, -1241.0316162, 1241.0316162

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3113394, upper bound: 1379.3150697
time: 1.04 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050628, upper bound: 1379.3104342
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -378.1729126, 1139.0440674, -382.3175964, 1150.1569824, -1528.3294678, 1521.3616943
1: -292.3612061, 757.0189819, -295.2808228, 764.3681030, -1056.7292480, 1052.2995605
2: -316.2933960, 719.6064453, -319.3530273, 726.3499146, -1042.6431885, 1038.9594727
3: -317.8445129, 922.1145020, -321.0665894, 931.2224121, -1249.0668945, 1243.1807861
4: -461.2918091, 779.7399292, -465.7024536, 787.0157471, -1248.3072510, 1245.4423828

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3113394, upper bound: 1379.3150697
time: 0.55 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050628, upper bound: 1379.3104342
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -382.3175964, 1150.1569824, -378.1729126, 1139.0440674, -1521.3616943, 1528.3294678
1: -295.2808228, 764.3681030, -292.3612061, 757.0189819, -1052.2995605, 1056.7292480
2: -319.3530273, 726.3499146, -316.2933960, 719.6064453, -1038.9594727, 1042.6431885
3: -321.0665894, 931.2224121, -317.8445129, 922.1145020, -1243.1807861, 1249.0668945
4: -465.7024536, 787.0157471, -461.2918091, 779.7399292, -1245.4423828, 1248.3072510

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050247, upper bound: 1379.3085845
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3049354, upper bound: 1379.3049354
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -382.3175964, 1150.1569824, -382.3175964, 1150.1569824, -1532.4746094, 1532.4746094
1: -295.2808228, 764.3681030, -295.2808228, 764.3681030, -1059.6489258, 1059.6489258
2: -319.3530273, 726.3499146, -319.3530273, 726.3499146, -1045.7028809, 1045.7028809
3: -321.0665894, 931.2224121, -321.0665894, 931.2224121, -1252.2889404, 1252.2889404
4: -465.7024536, 787.0157471, -465.7024536, 787.0157471, -1252.7182617, 1252.7182617

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_B1

### Relational analysis result of NS_A1_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3085845, upper bound: 1379.3050247
time: 0.61 seconds

## Relational analysis of NS_A1_B1_A2_B2_B2

### Relational analysis result of NS_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3049354, upper bound: 1379.3049354
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -378.1729126, 1139.0440674, -374.8723755, 1131.2322998, -1509.4049072, 1513.9165039
1: -292.3612061, 757.0189819, -289.8943787, 751.3854980, -1043.7465820, 1046.9132080
2: -316.2933960, 719.6064453, -313.6442871, 714.5298462, -1030.8231201, 1033.2507324
3: -317.8445129, 922.1145020, -315.4222412, 915.2698364, -1233.1143799, 1237.5366211
4: -461.2918091, 779.7399292, -457.7706299, 774.0721436, -1235.3634033, 1237.5104980

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3150511, upper bound: 1379.3113554
time: 0.64 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107604, upper bound: 1379.3050628
time: 0.83 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -382.3175964, 1150.1569824, -374.8723755, 1131.2322998, -1513.5499268, 1525.0292969
1: -295.2808228, 764.3681030, -289.8943787, 751.3854980, -1046.6662598, 1054.2624512
2: -319.3530273, 726.3499146, -313.6442871, 714.5298462, -1033.8828125, 1039.9941406
3: -321.0665894, 931.2224121, -315.4222412, 915.2698364, -1236.3363037, 1246.6446533
4: -465.7024536, 787.0157471, -457.7706299, 774.0721436, -1239.7744141, 1244.7862549

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3150511, upper bound: 1379.3113554
time: 0.58 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3107604, upper bound: 1379.3050628
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -378.1729126, 1139.0440674, -376.9593506, 1136.4855957, -1514.6582031, 1516.0034180
1: -292.3612061, 757.0189819, -291.3263855, 754.8334961, -1047.1945801, 1048.3453369
2: -316.2933960, 719.6064453, -315.1031189, 717.6030884, -1033.8964844, 1034.7095947
3: -317.8445129, 922.1145020, -317.0129089, 919.6062012, -1237.4505615, 1239.1271973
4: -461.2918091, 779.7399292, -459.8348999, 777.3976440, -1238.6889648, 1239.5747070

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3085852, upper bound: 1379.3050247
time: 0.88 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3088915, upper bound: 1379.3050145
time: 0.79 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -382.3175964, 1150.1569824, -376.9593506, 1136.4855957, -1518.8032227, 1527.1163330
1: -295.2808228, 764.3681030, -291.3263855, 754.8334961, -1050.1142578, 1055.6944580
2: -319.3530273, 726.3499146, -315.1031189, 717.6030884, -1036.9560547, 1041.4527588
3: -321.0665894, 931.2224121, -317.0129089, 919.6062012, -1240.6723633, 1248.2353516
4: -465.7024536, 787.0157471, -459.8348999, 777.3976440, -1243.1000977, 1246.8503418

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109663, upper bound: 1379.3112575
time: 0.70 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3088915, upper bound: 1379.3050145
time: 0.82 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -374.8723755, 1131.2322998, -378.1729126, 1139.0440674, -1513.9165039, 1509.4049072
1: -289.8943787, 751.3854980, -292.3612061, 757.0189819, -1046.9132080, 1043.7465820
2: -313.6442871, 714.5298462, -316.2933960, 719.6064453, -1033.2507324, 1030.8229980
3: -315.4222412, 915.2698364, -317.8445129, 922.1145020, -1237.5366211, 1233.1143799
4: -457.7706299, 774.0721436, -461.2918091, 779.7399292, -1237.5104980, 1235.3634033

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3113554, upper bound: 1379.3150511
time: 0.56 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050628, upper bound: 1379.3107604
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -374.8723755, 1131.2322998, -382.3175964, 1150.1569824, -1525.0292969, 1513.5499268
1: -289.8943787, 751.3854980, -295.2808228, 764.3681030, -1054.2624512, 1046.6662598
2: -313.6442871, 714.5298462, -319.3530273, 726.3499146, -1039.9941406, 1033.8828125
3: -315.4222412, 915.2698364, -321.0665894, 931.2224121, -1246.6446533, 1236.3363037
4: -457.7706299, 774.0721436, -465.7024536, 787.0157471, -1244.7862549, 1239.7744141

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3113554, upper bound: 1379.3150511
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050628, upper bound: 1379.3107604
time: 0.68 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -376.9593506, 1136.4855957, -378.1729126, 1139.0440674, -1516.0034180, 1514.6582031
1: -291.3263855, 754.8334961, -292.3612061, 757.0189819, -1048.3453369, 1047.1945801
2: -315.1031189, 717.6030884, -316.2933960, 719.6064453, -1034.7095947, 1033.8963623
3: -317.0129089, 919.6062012, -317.8445129, 922.1145020, -1239.1271973, 1237.4505615
4: -459.8348999, 777.3976440, -461.2918091, 779.7399292, -1239.5747070, 1238.6890869

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050247, upper bound: 1379.3085852
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3088916
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -376.9593506, 1136.4855957, -382.3175964, 1150.1569824, -1527.1163330, 1518.8032227
1: -291.3263855, 754.8334961, -295.2808228, 764.3681030, -1055.6944580, 1050.1142578
2: -315.1031189, 717.6030884, -319.3530273, 726.3499146, -1041.4527588, 1036.9560547
3: -317.0129089, 919.6062012, -321.0665894, 931.2224121, -1248.2353516, 1240.6723633
4: -459.8348999, 777.3976440, -465.7024536, 787.0157471, -1246.8503418, 1243.1000977

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112575, upper bound: 1379.3109663
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3088916
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -374.8723755, 1131.2322998, -374.8723755, 1131.2322998, -1506.1047363, 1506.1047363
1: -289.8943787, 751.3854980, -289.8943787, 751.3854980, -1041.2797852, 1041.2797852
2: -313.6442871, 714.5298462, -313.6442871, 714.5298462, -1028.1740723, 1028.1740723
3: -315.4222412, 915.2698364, -315.4222412, 915.2698364, -1230.6921387, 1230.6921387
4: -457.7706299, 774.0721436, -457.7706299, 774.0721436, -1231.8424072, 1231.8424072

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3139227
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110142, upper bound: 1379.3155545
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -374.8723755, 1131.2322998, -376.9593506, 1136.4855957, -1511.3579102, 1508.1916504
1: -289.8943787, 751.3854980, -291.3263855, 754.8334961, -1044.7277832, 1042.7119141
2: -313.6442871, 714.5298462, -315.1031189, 717.6030884, -1031.2473145, 1029.6326904
3: -315.4222412, 915.2698364, -317.0129089, 919.6062012, -1235.0283203, 1232.2827148
4: -457.7706299, 774.0721436, -459.8348999, 777.3976440, -1235.1680908, 1233.9066162

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3126401, upper bound: 1379.3158492
time: 0.90 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110142, upper bound: 1379.3155545
time: 0.78 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -376.9593506, 1136.4855957, -374.8723755, 1131.2322998, -1508.1916504, 1511.3579102
1: -291.3263855, 754.8334961, -289.8943787, 751.3854980, -1042.7119141, 1044.7277832
2: -315.1031189, 717.6030884, -313.6442871, 714.5298462, -1029.6326904, 1031.2473145
3: -317.0129089, 919.6062012, -315.4222412, 915.2698364, -1232.2827148, 1235.0283203
4: -459.8348999, 777.3976440, -457.7706299, 774.0721436, -1233.9066162, 1235.1680908

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3120980
time: 0.61 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -376.9593506, 1136.4855957, -376.9593506, 1136.4855957, -1513.4449463, 1513.4449463
1: -291.3263855, 754.8334961, -291.3263855, 754.8334961, -1046.1599121, 1046.1599121
2: -315.1031189, 717.6030884, -315.1031189, 717.6030884, -1032.7060547, 1032.7060547
3: -317.0129089, 919.6062012, -317.0129089, 919.6062012, -1236.6187744, 1236.6187744
4: -459.8348999, 777.3976440, -459.8348999, 777.3976440, -1237.2320557, 1237.2321777

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3120980
time: 1.06 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
time: 0.98 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.11 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3113394, upper bound: 1379.3150697
NS_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3050628, upper bound: 1379.3104342
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3113394, upper bound: 1379.3150697
NS_A1_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3050628, upper bound: 1379.3104342
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3050247, upper bound: 1379.3085845
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3049354, upper bound: 1379.3049354
NS_A1_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3085845, upper bound: 1379.3050247
NS_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3049354, upper bound: 1379.3049354
NS_A1_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3150511, upper bound: 1379.3113554
NS_A1_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3107604, upper bound: 1379.3050628
NS_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3150511, upper bound: 1379.3113554
NS_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3107604, upper bound: 1379.3050628
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3085852, upper bound: 1379.3050247
NS_A1_B2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3088915, upper bound: 1379.3050145
NS_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3109663, upper bound: 1379.3112575
NS_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3088915, upper bound: 1379.3050145
NS_A2_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3113554, upper bound: 1379.3150511
NS_A2_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3050628, upper bound: 1379.3107604
NS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3113554, upper bound: 1379.3150511
NS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3050628, upper bound: 1379.3107604
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3050247, upper bound: 1379.3085852
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3088916
NS_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3112575, upper bound: 1379.3109663
NS_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3088916
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3139227
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3110142, upper bound: 1379.3155545
NS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3126401, upper bound: 1379.3158492
NS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3110142, upper bound: 1379.3155545
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3120980
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3120980
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.11
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -362.7439270, 1091.0794678, -330.2405090, 993.6204834, -1356.3643799, 1421.3199463
1: -280.2392273, 725.0717163, -254.7778778, 659.3219604, -939.5611572, 979.8496094
2: -303.4712524, 689.0566406, -275.6751404, 626.4904785, -929.9617310, 964.7318115
3: -304.7832336, 883.4353027, -277.3078003, 803.4201050, -1108.2031250, 1160.7429199
4: -442.3126831, 746.7266846, -402.2677612, 678.8594360, -1121.1718750, 1148.9943848

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154950
time: 0.58 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154950
time: 0.66 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -374.6778564, 1127.6390381, -373.4217224, 1123.4768066, -1498.1546631, 1501.0607910
1: -289.5628052, 749.6141968, -288.5417175, 746.9279785, -1036.4907227, 1038.1558838
2: -313.3570862, 712.2893677, -312.2846069, 709.5980835, -1022.9552002, 1024.5738525
3: -314.8017273, 913.1320801, -313.6976013, 909.8775635, -1224.6793213, 1226.8297119
4: -456.9168701, 771.7606201, -455.3222961, 768.7992554, -1225.7159424, 1227.0828857

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154950
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154950
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -362.7439270, 1091.0794678, -336.7857361, 1012.2489014, -1374.9927979, 1427.8652344
1: -280.2392273, 725.0717163, -259.5390930, 671.5034180, -951.7426758, 984.6108398
2: -303.4712524, 689.0566406, -280.7144775, 637.8499756, -941.3212280, 969.7710571
3: -304.7832336, 883.4353027, -282.5847168, 818.5271606, -1123.3104248, 1166.0198975
4: -442.3126831, 746.7266846, -409.6542053, 691.1157227, -1133.4282227, 1156.3808594

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050587, upper bound: 1379.3098555
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050587, upper bound: 1379.3104342
time: 0.69 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -374.6778564, 1127.6390381, -378.1353760, 1136.2581787, -1510.9360352, 1505.7744141
1: -289.5628052, 749.6141968, -291.8923645, 755.3835449, -1044.9462891, 1041.5065918
2: -313.3570862, 712.2893677, -315.8110352, 717.4314575, -1030.7885742, 1028.1003418
3: -314.8017273, 913.1320801, -317.3814087, 920.3523560, -1235.1540527, 1230.5134277
4: -456.9168701, 771.7606201, -460.3802185, 777.1884155, -1234.1052246, 1232.1407471

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050587, upper bound: 1379.3098555
time: 0.67 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050587, upper bound: 1379.3104342
time: 0.75 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -336.7857361, 1012.2489014, -362.7439270, 1091.0794678, -1427.8652344, 1374.9927979
1: -259.5390930, 671.5034180, -280.2392273, 725.0717163, -984.6108398, 951.7426758
2: -280.7144775, 637.8499756, -303.4712524, 689.0566406, -969.7710571, 941.3212280
3: -282.5847168, 818.5271606, -304.7832336, 883.4353027, -1166.0198975, 1123.3103027
4: -409.6542053, 691.1157227, -442.3126831, 746.7266846, -1156.3808594, 1133.4282227

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3098555, upper bound: 1379.3050587
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3098555, upper bound: 1379.3050628
time: 0.60 seconds

## BFS NS instance: NS_A1_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -365.9288025, 1099.1326904, -336.7857361, 1012.2489014, -1378.1777344, 1435.9184570
1: -282.4515991, 730.4550171, -259.5390930, 671.5034180, -953.9550171, 989.9941406
2: -305.7623291, 693.8624878, -280.7144775, 637.8499756, -943.6123047, 974.5769653
3: -307.2194824, 890.0911255, -282.5847168, 818.5271606, -1125.7464600, 1172.6757812
4: -445.5678406, 751.9313965, -409.6542053, 691.1157227, -1136.6833496, 1161.5855713

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3049354, upper bound: 1379.3049354
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3049354, upper bound: 1379.3049354
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1

### Backsubstitution after applying NS history:
0: -330.2405090, 993.6204834, -358.9984741, 1081.4368896, -1411.6773682, 1352.6188965
1: -254.7778778, 659.3219604, -277.3256531, 718.1442261, -972.9221191, 936.6475220
2: -275.6751404, 626.4904785, -300.3528442, 682.7649536, -958.4400635, 926.8431396
3: -277.3078003, 803.4201050, -301.9778137, 875.0473633, -1152.3549805, 1105.3979492
4: -402.2677612, 678.8594360, -438.0067444, 739.7766113, -1142.0440674, 1116.8658447

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
time: 0.75 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2

### Backsubstitution after applying NS history:
0: -373.4217224, 1123.4768066, -371.7628784, 1121.0468750, -1494.4686279, 1495.2396240
1: -288.5417175, 746.9279785, -287.4142151, 744.7874756, -1033.3291016, 1034.3421631
2: -312.2846069, 709.5980835, -311.0434875, 708.0032349, -1020.2878418, 1020.6416016
3: -313.6976013, 909.8775635, -312.7127686, 907.2399902, -1220.9376221, 1222.5903320
4: -455.3222961, 768.7992554, -453.8881531, 766.9556885, -1222.2779541, 1222.6872559

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
time: 0.64 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1

### Backsubstitution after applying NS history:
0: -336.7857361, 1012.2489014, -358.9984741, 1081.4368896, -1418.2226562, 1371.2473145
1: -259.5390930, 671.5034180, -277.3256531, 718.1442261, -977.6833496, 948.8291016
2: -280.7144775, 637.8499756, -300.3528442, 682.7649536, -963.4793701, 938.2026978
3: -282.5847168, 818.5271606, -301.9778137, 875.0473633, -1157.6319580, 1120.5050049
4: -409.6542053, 691.1157227, -438.0067444, 739.7766113, -1149.4302979, 1129.1221924

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_A1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3097083, upper bound: 1379.3050523
time: 0.69 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3097083, upper bound: 1379.3050628
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2

### Backsubstitution after applying NS history:
0: -378.1353760, 1136.2581787, -371.7628784, 1121.0468750, -1499.1822510, 1508.0208740
1: -291.8923645, 755.3835449, -287.4142151, 744.7874756, -1036.6798096, 1042.7977295
2: -315.8110352, 717.4314575, -311.0434875, 708.0032349, -1023.8142700, 1028.4748535
3: -317.3814087, 920.3523560, -312.7127686, 907.2399902, -1224.6213379, 1233.0651855
4: -460.3802185, 777.1884155, -453.8881531, 766.9556885, -1227.3359375, 1231.0765381

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B1_A2_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3097083, upper bound: 1379.3050523
time: 0.60 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3097083, upper bound: 1379.3050628
time: 0.66 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -362.7439270, 1091.0794678, -325.6672668, 979.7856445, -1342.5295410, 1416.7467041
1: -280.2392273, 725.0717163, -251.0235138, 649.6229858, -929.8621826, 976.0952148
2: -303.4712524, 689.0566406, -271.6387329, 617.2684937, -920.7397461, 960.6953735
3: -304.7832336, 883.4353027, -273.6169739, 791.9397583, -1096.7225342, 1157.0522461
4: -442.3126831, 746.7266846, -396.4029541, 668.7114868, -1111.0241699, 1143.1296387

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3149153
time: 0.59 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3150299
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -374.6778564, 1127.6390381, -373.0659790, 1123.5406494, -1498.2185059, 1500.7050781
1: -289.5628052, 749.6141968, -288.1848450, 746.4497070, -1036.0123291, 1037.7990723
2: -313.3570862, 712.2893677, -311.8339233, 709.3007812, -1022.6578369, 1024.1232910
3: -314.8017273, 913.1320801, -313.5817566, 909.4652710, -1224.2669678, 1226.7138672
4: -456.9168701, 771.7606201, -454.9142456, 768.2630005, -1225.1799316, 1226.6748047

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3149152
time: 1.00 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3150299
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1

### Backsubstitution after applying NS history:
0: -336.7857361, 1012.2489014, -360.6119080, 1084.8586426, -1421.6444092, 1372.8608398
1: -259.5390930, 671.5034180, -278.3677979, 720.4926147, -980.0317383, 949.8712158
2: -280.7144775, 637.8499756, -301.3521118, 684.6701660, -965.3845825, 939.2019653
3: -282.5847168, 818.5271606, -303.1584167, 878.0120239, -1160.5966797, 1121.6855469
4: -409.6542053, 691.1157227, -439.3654480, 741.8388062, -1151.4926758, 1130.4810791

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3085494, upper bound: 1379.3050145
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3085494, upper bound: 1379.3050145
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2

### Backsubstitution after applying NS history:
0: -378.1353760, 1136.2581787, -374.3325806, 1127.7530518, -1505.8884277, 1510.5908203
1: -291.8923645, 755.3835449, -289.2108765, 749.1699219, -1041.0622559, 1044.5942383
2: -315.8110352, 717.4314575, -312.8998108, 711.9945679, -1027.8056641, 1030.3312988
3: -317.3814087, 920.3523560, -314.7020569, 912.7528076, -1230.1342773, 1235.0543213
4: -460.3802185, 777.1884155, -456.5171814, 771.2510986, -1231.6309814, 1233.7054443

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_B2_A2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3085494, upper bound: 1379.3050145
time: 0.66 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3085494, upper bound: 1379.3050145
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -358.9984741, 1081.4368896, -330.2405090, 993.6204834, -1352.6188965, 1411.6773682
1: -277.3256531, 718.1442261, -254.7778778, 659.3219604, -936.6475220, 972.9221191
2: -300.3528442, 682.7649536, -275.6751404, 626.4904785, -926.8431396, 958.4400635
3: -301.9778137, 875.0473633, -277.3078003, 803.4201050, -1105.3979492, 1152.3551025
4: -438.0067444, 739.7766113, -402.2677612, 678.8594360, -1116.8658447, 1142.0440674

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3156500
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3168903
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -371.7628784, 1121.0468750, -373.4217224, 1123.4768066, -1495.2396240, 1494.4686279
1: -287.4142151, 744.7874756, -288.5417175, 746.9279785, -1034.3421631, 1033.3291016
2: -311.0434875, 708.0032349, -312.2846069, 709.5980835, -1020.6416016, 1020.2878418
3: -312.7127686, 907.2399902, -313.6976013, 909.8775635, -1222.5903320, 1220.9376221
4: -453.8881531, 766.9556885, -455.3222961, 768.7992554, -1222.6872559, 1222.2779541

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3156500
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3171966
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -358.9984741, 1081.4368896, -336.7857361, 1012.2489014, -1371.2473145, 1418.2226562
1: -277.3256531, 718.1442261, -259.5390930, 671.5034180, -948.8291016, 977.6833496
2: -300.3528442, 682.7649536, -280.7144775, 637.8499756, -938.2026978, 963.4793701
3: -301.9778137, 875.0473633, -282.5847168, 818.5271606, -1120.5050049, 1157.6320801
4: -438.0067444, 739.7766113, -409.6542053, 691.1157227, -1129.1220703, 1149.4302979

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050523, upper bound: 1379.3097083
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050523, upper bound: 1379.3107604
time: 0.79 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -371.7628784, 1121.0468750, -378.1353760, 1136.2581787, -1508.0208740, 1499.1822510
1: -287.4142151, 744.7874756, -291.8923645, 755.3835449, -1042.7977295, 1036.6798096
2: -311.0434875, 708.0032349, -315.8110352, 717.4314575, -1028.4748535, 1023.8142700
3: -312.7127686, 907.2399902, -317.3814087, 920.3523560, -1233.0651855, 1224.6213379
4: -453.8881531, 766.9556885, -460.3802185, 777.1884155, -1231.0765381, 1227.3359375

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050523, upper bound: 1379.3097083
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050523, upper bound: 1379.3107604
time: 0.74 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -325.6672668, 979.7856445, -362.7439270, 1091.0794678, -1416.7467041, 1342.5295410
1: -251.0235138, 649.6229858, -280.2392273, 725.0717163, -976.0952148, 929.8621826
2: -271.6387329, 617.2684937, -303.4712524, 689.0566406, -960.6953735, 920.7396851
3: -273.6169739, 791.9397583, -304.7832336, 883.4353027, -1157.0522461, 1096.7225342
4: -396.4029541, 668.7114868, -442.3126831, 746.7266846, -1143.1296387, 1111.0241699

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -373.0659790, 1123.5406494, -374.6778564, 1127.6390381, -1500.7050781, 1498.2185059
1: -288.1848450, 746.4497070, -289.5628052, 749.6141968, -1037.7990723, 1036.0123291
2: -311.8339233, 709.3007812, -313.3570862, 712.2893677, -1024.1232910, 1022.6578369
3: -313.5817566, 909.4652710, -314.8017273, 913.1320801, -1226.7138672, 1224.2669678
4: -454.9142456, 768.2630005, -456.9168701, 771.7606201, -1226.6748047, 1225.1799316

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
time: 0.76 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -360.6119080, 1084.8586426, -336.7857361, 1012.2489014, -1372.8608398, 1421.6444092
1: -278.3677979, 720.4926147, -259.5390930, 671.5034180, -949.8712158, 980.0317383
2: -301.3521118, 684.6701660, -280.7144775, 637.8499756, -939.2019653, 965.3845825
3: -303.1584167, 878.0120239, -282.5847168, 818.5271606, -1121.6855469, 1160.5966797
4: -439.3654480, 741.8388062, -409.6542053, 691.1157227, -1130.4809570, 1151.4926758

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3085494
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3088916
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -374.3325806, 1127.7530518, -378.1353760, 1136.2581787, -1510.5908203, 1505.8884277
1: -289.2108765, 749.1699219, -291.8923645, 755.3835449, -1044.5943604, 1041.0622559
2: -312.8998108, 711.9945679, -315.8110352, 717.4314575, -1030.3312988, 1027.8056641
3: -314.7020569, 912.7528076, -317.3814087, 920.3523560, -1235.0543213, 1230.1342773
4: -456.5171814, 771.2510986, -460.3802185, 777.1884155, -1233.7054443, 1231.6309814

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3085494
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3088916
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -321.8771362, 969.6198120, -358.9984741, 1081.4368896, -1403.3139648, 1328.6182861
1: -248.3694458, 643.0812988, -277.3256531, 718.1442261, -966.5136719, 920.4069824
2: -268.8417664, 611.3493652, -300.3528442, 682.7649536, -951.6066284, 911.7021484
3: -270.6230164, 783.6445312, -301.9778137, 875.0473633, -1145.6702881, 1085.6223145
4: -392.3271790, 662.3067017, -438.0067444, 739.7766113, -1132.1031494, 1100.3134766

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156701, upper bound: 1379.3156550
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3156550
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -370.2542419, 1116.0351562, -371.7628784, 1121.0468750, -1491.3011475, 1487.7977295
1: -286.2065125, 741.5526733, -287.4142151, 744.7874756, -1030.9937744, 1028.9669189
2: -309.7839050, 704.7974854, -311.0434875, 708.0032349, -1017.7871094, 1015.8409424
3: -311.3895874, 903.3079834, -312.7127686, 907.2399902, -1218.6295166, 1216.0207520
4: -451.9976501, 763.4550781, -453.8881531, 766.9556885, -1218.9533691, 1217.3431396

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156702, upper bound: 1379.3169024
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156702, upper bound: 1379.3188852
time: 0.90 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -358.9984741, 1081.4368896, -325.6672668, 979.7856445, -1338.7841797, 1407.1041260
1: -277.3256531, 718.1442261, -251.0235138, 649.6229858, -926.9486084, 969.1677246
2: -300.3528442, 682.7649536, -271.6387329, 617.2684937, -917.6211548, 954.4036865
3: -301.9778137, 875.0473633, -273.6169739, 791.9397583, -1093.9173584, 1148.6643066
4: -438.0067444, 739.7766113, -396.4029541, 668.7114868, -1106.7181396, 1136.1793213

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3139227
time: 0.68 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3155545
time: 0.62 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -371.7628784, 1121.0468750, -373.0659790, 1123.5406494, -1495.3034668, 1494.1127930
1: -287.4142151, 744.7874756, -288.1848450, 746.4497070, -1033.8637695, 1032.9721680
2: -311.0434875, 708.0032349, -311.8339233, 709.3007812, -1020.3442383, 1019.8371582
3: -312.7127686, 907.2399902, -313.5817566, 909.4652710, -1222.1779785, 1220.8217773
4: -453.8881531, 766.9556885, -454.9142456, 768.2630005, -1222.1511230, 1221.8698730

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3139227
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3155545
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -325.6672668, 979.7856445, -358.9984741, 1081.4368896, -1407.1041260, 1338.7841797
1: -251.0235138, 649.6229858, -277.3256531, 718.1442261, -969.1677246, 926.9486084
2: -271.6387329, 617.2684937, -300.3528442, 682.7649536, -954.4036865, 917.6211548
3: -273.6169739, 791.9397583, -301.9778137, 875.0473633, -1148.6643066, 1093.9172363
4: -396.4029541, 668.7114868, -438.0067444, 739.7766113, -1136.1793213, 1106.7181396

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3148869, upper bound: 1379.3110357
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3148869, upper bound: 1379.3110357
time: 0.76 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -373.0659790, 1123.5406494, -371.7628784, 1121.0468750, -1494.1127930, 1495.3034668
1: -288.1848450, 746.4497070, -287.4142151, 744.7874756, -1032.9722900, 1033.8637695
2: -311.8339233, 709.3007812, -311.0434875, 708.0032349, -1019.8371582, 1020.3442383
3: -313.5817566, 909.4652710, -312.7127686, 907.2399902, -1220.8217773, 1222.1779785
4: -454.9142456, 768.2630005, -453.8881531, 766.9556885, -1221.8698730, 1222.1511230

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3148869, upper bound: 1379.3110356
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3148869, upper bound: 1379.3110357
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -325.6672668, 979.7856445, -360.6119080, 1084.8586426, -1410.5258789, 1340.3975830
1: -251.0235138, 649.6229858, -278.3677979, 720.4926147, -971.5161133, 927.9907837
2: -271.6387329, 617.2684937, -301.3521118, 684.6701660, -956.3088989, 918.6204834
3: -273.6169739, 791.9397583, -303.1584167, 878.0120239, -1151.6290283, 1095.0980225
4: -396.4029541, 668.7114868, -439.3654480, 741.8388062, -1138.2415771, 1108.0769043

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -373.0659790, 1123.5406494, -374.3325806, 1127.7530518, -1500.8189697, 1497.8732910
1: -288.1848450, 746.4497070, -289.2108765, 749.1699219, -1037.3547363, 1035.6604004
2: -311.8339233, 709.3007812, -312.8998108, 711.9945679, -1023.8283691, 1022.2005615
3: -313.5817566, 909.4652710, -314.7020569, 912.7528076, -1226.3344727, 1224.1672363
4: -454.9142456, 768.2630005, -456.5171814, 771.2510986, -1226.1652832, 1224.7801514

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
time: 0.79 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.26 seconds
NS_A1_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154950
NS_A1_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154950
NS_A1_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154950
NS_A1_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154950
NS_A1_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050587, upper bound: 1379.3098555
NS_A1_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050587, upper bound: 1379.3104342
NS_A1_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050587, upper bound: 1379.3098555
NS_A1_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050587, upper bound: 1379.3104342
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3098555, upper bound: 1379.3050587
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3098555, upper bound: 1379.3050628
NS_A1_B1_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3049354, upper bound: 1379.3049354
NS_A1_B1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3049354, upper bound: 1379.3049354
NS_A1_B2_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
NS_A1_B2_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
NS_A1_B2_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
NS_A1_B2_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
NS_A1_B2_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3097083, upper bound: 1379.3050523
NS_A1_B2_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3097083, upper bound: 1379.3050628
NS_A1_B2_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3097083, upper bound: 1379.3050523
NS_A1_B2_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3097083, upper bound: 1379.3050628
NS_A1_B2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3149153
NS_A1_B2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3150299
NS_A1_B2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3149152
NS_A1_B2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3150299
NS_A1_B2_B2_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3085494, upper bound: 1379.3050145
NS_A1_B2_B2_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3085494, upper bound: 1379.3050145
NS_A1_B2_B2_A2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3085494, upper bound: 1379.3050145
NS_A1_B2_B2_A2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3085494, upper bound: 1379.3050145
NS_A2_B1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3156500
NS_A2_B1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3168903
NS_A2_B1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3156500
NS_A2_B1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3171966
NS_A2_B1_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050523, upper bound: 1379.3097083
NS_A2_B1_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050523, upper bound: 1379.3107604
NS_A2_B1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050523, upper bound: 1379.3097083
NS_A2_B1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050523, upper bound: 1379.3107604
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
NS_A2_B1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3085494
NS_A2_B1_A2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3088916
NS_A2_B1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3085494
NS_A2_B1_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3088916
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3156701, upper bound: 1379.3156550
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3156550
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3156702, upper bound: 1379.3169024
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3156702, upper bound: 1379.3188852
NS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3139227
NS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3155545
NS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3139227
NS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3155545
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3148869, upper bound: 1379.3110357
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3148869, upper bound: 1379.3110357
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3148869, upper bound: 1379.3110356
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3148869, upper bound: 1379.3110357
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 3.26
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514

## BFS NS instance: NS_A1_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -330.2405090, 993.6204834, -330.2405090, 993.6204834, -1323.8609619, 1323.8609619
1: -254.7778778, 659.3219604, -254.7778778, 659.3219604, -914.0998535, 914.0998535
2: -275.6751404, 626.4904785, -275.6751404, 626.4904785, -902.1655884, 902.1655884
3: -277.3078003, 803.4201050, -277.3078003, 803.4201050, -1080.7277832, 1080.7277832
4: -402.2677612, 678.8594360, -402.2677612, 678.8594360, -1081.1269531, 1081.1269531

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3132005, upper bound: 1379.3101000
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156111, upper bound: 1379.3155131
time: 0.61 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -373.4217224, 1123.4768066, -330.2405090, 993.6204834, -1367.0422363, 1453.7172852
1: -288.5417175, 746.9279785, -254.7778778, 659.3219604, -947.8636475, 1001.7058105
2: -312.2846069, 709.5980835, -275.6751404, 626.4904785, -938.7750244, 985.2731934
3: -313.6976013, 909.8775635, -277.3078003, 803.4201050, -1117.1176758, 1187.1853027
4: -455.3222961, 768.7992554, -402.2677612, 678.8594360, -1134.1815186, 1171.0668945

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3101000, upper bound: 1379.3134649
time: 0.80 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156111, upper bound: 1379.3155131
time: 0.63 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -330.2405090, 993.6204834, -373.4217224, 1123.4768066, -1453.7172852, 1367.0422363
1: -254.7778778, 659.3219604, -288.5417175, 746.9279785, -1001.7058105, 947.8636475
2: -275.6751404, 626.4904785, -312.2846069, 709.5980835, -985.2731934, 938.7750244
3: -277.3078003, 803.4201050, -313.6976013, 909.8775635, -1187.1853027, 1117.1176758
4: -402.2677612, 678.8594360, -455.3222961, 768.7992554, -1171.0668945, 1134.1815186

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3132005, upper bound: 1379.3101000
time: 0.69 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154949
time: 0.67 seconds

## BFS NS instance: NS_A1_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -373.4217224, 1123.4768066, -373.4217224, 1123.4768066, -1496.8985596, 1496.8985596
1: -288.5417175, 746.9279785, -288.5417175, 746.9279785, -1035.4696045, 1035.4696045
2: -312.2846069, 709.5980835, -312.2846069, 709.5980835, -1021.8826904, 1021.8826904
3: -313.6976013, 909.8775635, -313.6976013, 909.8775635, -1223.5751953, 1223.5751953
4: -455.3222961, 768.7992554, -455.3222961, 768.7992554, -1224.1213379, 1224.1213379

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3132005, upper bound: 1379.3123640
time: 0.71 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154950
time: 0.64 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -330.2405090, 993.6204834, -336.7857361, 1012.2489014, -1342.4893799, 1330.4062500
1: -254.7778778, 659.3219604, -259.5390930, 671.5034180, -926.2813110, 918.8610840
2: -275.6751404, 626.4904785, -280.7144775, 637.8499756, -913.5251465, 907.2048340
3: -277.3078003, 803.4201050, -282.5847168, 818.5271606, -1095.8349609, 1086.0047607
4: -402.2677612, 678.8594360, -409.6542053, 691.1157227, -1093.3834229, 1088.5133057

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3068580, upper bound: 1379.3117708
time: 0.66 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3068580, upper bound: 1379.3150697
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -373.4217224, 1123.4768066, -336.7857361, 1012.2489014, -1385.6706543, 1460.2625732
1: -288.5417175, 746.9279785, -259.5390930, 671.5034180, -960.0451660, 1006.4670410
2: -312.2846069, 709.5980835, -280.7144775, 637.8499756, -950.1345825, 990.3125000
3: -313.6976013, 909.8775635, -282.5847168, 818.5271606, -1132.2247314, 1192.4622803
4: -455.3222961, 768.7992554, -409.6542053, 691.1157227, -1146.4379883, 1178.4531250

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3068580, upper bound: 1379.3121291
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3113394, upper bound: 1379.3150697
time: 0.56 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -330.2405090, 993.6204834, -378.1353760, 1136.2581787, -1466.4986572, 1371.7558594
1: -254.7778778, 659.3219604, -291.8923645, 755.3835449, -1010.1614380, 951.2143555
2: -275.6751404, 626.4904785, -315.8110352, 717.4314575, -993.1065674, 942.3015137
3: -277.3078003, 803.4201050, -317.3814087, 920.3523560, -1197.6601562, 1120.8015137
4: -402.2677612, 678.8594360, -460.3802185, 777.1884155, -1179.4561768, 1139.2392578

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3046254, upper bound: 1379.3088612
time: 0.63 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3046160, upper bound: 1379.3085028
time: 0.73 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -373.4217224, 1123.4768066, -378.1353760, 1136.2581787, -1509.6799316, 1501.6121826
1: -288.5417175, 746.9279785, -291.8923645, 755.3835449, -1043.9250488, 1038.8203125
2: -312.2846069, 709.5980835, -315.8110352, 717.4314575, -1029.7158203, 1025.4090576
3: -313.6976013, 909.8775635, -317.3814087, 920.3523560, -1234.0499268, 1227.2590332
4: -455.3222961, 768.7992554, -460.3802185, 777.1884155, -1232.5106201, 1229.1791992

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3046253, upper bound: 1379.3093303
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3046159, upper bound: 1379.3085028
time: 0.79 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -336.7857361, 1012.2489014, -330.2405090, 993.6204834, -1330.4062500, 1342.4893799
1: -259.5390930, 671.5034180, -254.7778778, 659.3219604, -918.8610229, 926.2813110
2: -280.7144775, 637.8499756, -275.6751404, 626.4904785, -907.2048340, 913.5251465
3: -282.5847168, 818.5271606, -277.3078003, 803.4201050, -1086.0047607, 1095.8349609
4: -409.6542053, 691.1157227, -402.2677612, 678.8594360, -1088.5133057, 1093.3834229

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3117708, upper bound: 1379.3068580
time: 0.75 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3150697, upper bound: 1379.3113394
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -336.7857361, 1012.2489014, -373.4217224, 1123.4768066, -1460.2625732, 1385.6706543
1: -259.5390930, 671.5034180, -288.5417175, 746.9279785, -1006.4670410, 960.0451660
2: -280.7144775, 637.8499756, -312.2846069, 709.5980835, -990.3125000, 950.1345825
3: -282.5847168, 818.5271606, -313.6976013, 909.8775635, -1192.4622803, 1132.2247314
4: -409.6542053, 691.1157227, -455.3222961, 768.7992554, -1178.4531250, 1146.4379883

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3117708, upper bound: 1379.3068584
time: 0.68 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3150697, upper bound: 1379.3113394
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -330.2405090, 993.6204834, -321.8771362, 969.6198120, -1299.8603516, 1315.4975586
1: -254.7778778, 659.3219604, -248.3694458, 643.0812988, -897.8591919, 907.6913452
2: -275.6751404, 626.4904785, -268.8417664, 611.3493652, -887.0245361, 895.3320923
3: -277.3078003, 803.4201050, -270.6230164, 783.6445312, -1060.9523926, 1074.0429688
4: -402.2677612, 678.8594360, -392.3271790, 662.3067017, -1064.5744629, 1071.1861572

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_A1_A1_B1_B1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3097324, upper bound: 1379.3131531
time: 0.77 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B1_B2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156667, upper bound: 1379.3156312
time: 0.84 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -330.2405090, 993.6204834, -370.2542419, 1116.0351562, -1446.2756348, 1363.8747559
1: -254.7778778, 659.3219604, -286.2065125, 741.5526733, -996.3305664, 945.5284424
2: -275.6751404, 626.4904785, -309.7839050, 704.7974854, -980.4726562, 936.2744141
3: -277.3078003, 803.4201050, -311.3895874, 903.3079834, -1180.6157227, 1114.8095703
4: -402.2677612, 678.8594360, -451.9976501, 763.4550781, -1165.7227783, 1130.8568115

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3131281, upper bound: 1379.3100999
time: 0.65 seconds

## Relational analysis of NS_A1_B2_B1_A1_A1_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156667, upper bound: 1379.3156312
time: 0.76 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -373.4217224, 1123.4768066, -321.8771362, 969.6198120, -1343.0415039, 1445.3540039
1: -288.5417175, 746.9279785, -248.3694458, 643.0812988, -931.6230469, 995.2973633
2: -312.2846069, 709.5980835, -268.8417664, 611.3493652, -923.6339722, 978.4397583
3: -313.6976013, 909.8775635, -270.6230164, 783.6445312, -1097.3421631, 1180.5006104
4: -455.3222961, 768.7992554, -392.3271790, 662.3067017, -1117.6290283, 1161.1259766

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_A1_A2_B1_B1

### Relational analysis result of NS_A1_B2_B1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3097324, upper bound: 1379.3134207
time: 0.91 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_B1_B2

### Relational analysis result of NS_A1_B2_B1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_B1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -373.4217224, 1123.4768066, -370.2542419, 1116.0351562, -1489.4569092, 1493.7310791
1: -288.5417175, 746.9279785, -286.2065125, 741.5526733, -1030.0943604, 1033.1342773
2: -312.2846069, 709.5980835, -309.7839050, 704.7974854, -1017.0820923, 1019.3819580
3: -313.6976013, 909.8775635, -311.3895874, 903.3079834, -1217.0056152, 1221.2670898
4: -455.3222961, 768.7992554, -451.9976501, 763.4550781, -1218.7772217, 1220.7966309

Time for backsubstitution: 2.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B1_A1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B1_A1_A2_B2_A1

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
time: 0.75 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2_B2_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3150342
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -336.7857361, 1012.2489014, -321.8771362, 969.6198120, -1306.4055176, 1334.1259766
1: -259.5390930, 671.5034180, -248.3694458, 643.0812988, -902.6203613, 919.8728638
2: -280.7144775, 637.8499756, -268.8417664, 611.3493652, -892.0638428, 906.6916504
3: -282.5847168, 818.5271606, -270.6230164, 783.6445312, -1066.2292480, 1089.1501465
4: -409.6542053, 691.1157227, -392.3271790, 662.3067017, -1071.9609375, 1083.4423828

Time for backsubstitution: 2.03 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3116870, upper bound: 1379.3068551
time: 1.02 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3150511, upper bound: 1379.3113554
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -336.7857361, 1012.2489014, -370.2542419, 1116.0351562, -1452.8209229, 1382.5031738
1: -259.5390930, 671.5034180, -286.2065125, 741.5526733, -1001.0917969, 957.7099609
2: -280.7144775, 637.8499756, -309.7839050, 704.7974854, -985.5119019, 947.6339111
3: -282.5847168, 818.5271606, -311.3895874, 903.3079834, -1185.8925781, 1129.9167480
4: -409.6542053, 691.1157227, -451.9976501, 763.4550781, -1173.1090088, 1143.1132812

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3146160, upper bound: 1379.3103479
time: 0.66 seconds

## Relational analysis of NS_A1_B2_B1_A2_A1_B2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3082791, upper bound: 1379.3102118
time: 0.65 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B1

### Backsubstitution after applying NS history:
0: -378.1353760, 1136.2581787, -321.8771362, 969.6198120, -1347.7551270, 1458.1352539
1: -291.8923645, 755.3835449, -248.3694458, 643.0812988, -934.9736328, 1003.7529907
2: -315.8110352, 717.4314575, -268.8417664, 611.3493652, -927.1604004, 986.2731323
3: -317.3814087, 920.3523560, -270.6230164, 783.6445312, -1101.0258789, 1190.9753418
4: -460.3802185, 777.1884155, -392.3271790, 662.3067017, -1122.6868896, 1169.5152588

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3086425, upper bound: 1379.3046175
time: 0.63 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3081297, upper bound: 1379.3046037
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -378.1353760, 1136.2581787, -370.2542419, 1116.0351562, -1494.1705322, 1506.5123291
1: -291.8923645, 755.3835449, -286.2065125, 741.5526733, -1033.4450684, 1041.5898438
2: -315.8110352, 717.4314575, -309.7839050, 704.7974854, -1020.6085205, 1027.2152100
3: -317.3814087, 920.3523560, -311.3895874, 903.3079834, -1220.6894531, 1231.7419434
4: -460.3802185, 777.1884155, -451.9976501, 763.4550781, -1223.8350830, 1229.1860352

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3086425, upper bound: 1379.3046176
time: 0.81 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_B1_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3081297, upper bound: 1379.3046037
time: 0.73 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -330.2405090, 993.6204834, -325.6672668, 979.7856445, -1310.0261230, 1319.2877197
1: -254.7778778, 659.3219604, -251.0235138, 649.6229858, -904.4008789, 910.3454590
2: -275.6751404, 626.4904785, -271.6387329, 617.2684937, -892.9436035, 898.1292114
3: -277.3078003, 803.4201050, -273.6169739, 791.9397583, -1069.2471924, 1077.0371094
4: -402.2677612, 678.8594360, -396.4029541, 668.7114868, -1070.9792480, 1075.2622070

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3086637, upper bound: 1379.3129919
time: 0.75 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3126465, upper bound: 1379.3154650
time: 0.69 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -373.4217224, 1123.4768066, -325.6672668, 979.7856445, -1353.2073975, 1449.1440430
1: -288.5417175, 746.9279785, -251.0235138, 649.6229858, -938.1646729, 997.9514771
2: -312.2846069, 709.5980835, -271.6387329, 617.2684937, -929.5530396, 981.2368164
3: -313.6976013, 909.8775635, -273.6169739, 791.9397583, -1105.6372070, 1183.4945068
4: -455.3222961, 768.7992554, -396.4029541, 668.7114868, -1124.0338135, 1165.2021484

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3086637, upper bound: 1379.3132686
time: 1.00 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3126466, upper bound: 1379.3154652
time: 0.72 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -330.2405090, 993.6204834, -373.0659790, 1123.5406494, -1453.7811279, 1366.6864014
1: -254.7778778, 659.3219604, -288.1848450, 746.4497070, -1001.2276001, 947.5066528
2: -275.6751404, 626.4904785, -311.8339233, 709.3007812, -984.9759521, 938.3244019
3: -277.3078003, 803.4201050, -313.5817566, 909.4652710, -1186.7730713, 1117.0018311
4: -402.2677612, 678.8594360, -454.9142456, 768.2630005, -1170.5307617, 1133.7735596

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109339, upper bound: 1379.3100484
time: 0.84 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3149152
time: 0.67 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -373.4217224, 1123.4768066, -373.0659790, 1123.5406494, -1496.9624023, 1496.5427246
1: -288.5417175, 746.9279785, -288.1848450, 746.4497070, -1034.9912109, 1035.1127930
2: -312.2846069, 709.5980835, -311.8339233, 709.3007812, -1021.5853882, 1021.4320068
3: -313.6976013, 909.8775635, -313.5817566, 909.4652710, -1223.1628418, 1223.4593506
4: -455.3222961, 768.7992554, -454.9142456, 768.2630005, -1223.5853271, 1223.7133789

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3086637, upper bound: 1379.3140796
time: 0.74 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3150299
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -336.7857361, 1012.2489014, -325.6672668, 979.7856445, -1316.5714111, 1337.9161377
1: -259.5390930, 671.5034180, -251.0235138, 649.6229858, -909.1621094, 922.5269165
2: -280.7144775, 637.8499756, -271.6387329, 617.2684937, -897.9828491, 909.4887085
3: -282.5847168, 818.5271606, -273.6169739, 791.9397583, -1074.5241699, 1092.1441650
4: -409.6542053, 691.1157227, -396.4029541, 668.7114868, -1078.3656006, 1087.5185547

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3086323, upper bound: 1379.3112048
time: 0.82 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109663, upper bound: 1379.3112575
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -336.7857361, 1012.2489014, -373.0659790, 1123.5406494, -1460.3264160, 1385.3148193
1: -259.5390930, 671.5034180, -288.1848450, 746.4497070, -1005.9887695, 959.6881714
2: -280.7144775, 637.8499756, -311.8339233, 709.3007812, -990.0151978, 949.6838989
3: -282.5847168, 818.5271606, -313.5817566, 909.4652710, -1192.0499268, 1132.1088867
4: -409.6542053, 691.1157227, -454.9142456, 768.2630005, -1177.9172363, 1146.0299072

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3102545, upper bound: 1379.3068165
time: 1.05 seconds

## Relational analysis of NS_A1_B2_B2_A2_A1_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109663, upper bound: 1379.3112575
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -378.1353760, 1136.2581787, -325.6672668, 979.7856445, -1357.9210205, 1461.9254150
1: -291.8923645, 755.3835449, -251.0235138, 649.6229858, -941.5153809, 1006.4070435
2: -315.8110352, 717.4314575, -271.6387329, 617.2684937, -933.0795288, 989.0701904
3: -317.3814087, 920.3523560, -273.6169739, 791.9397583, -1109.3210449, 1193.9693604
4: -460.3802185, 777.1884155, -396.4029541, 668.7114868, -1129.0916748, 1173.5913086

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 35

### Candidate
type: B, layer: 1, pos: 11

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 34

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 34

### Candidate
type: A, layer: 1, pos: 11

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3036158, upper bound: 1379.3048301
time: 0.68 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B1_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3085494, upper bound: 1379.3050145
time: 0.70 seconds

## BFS NS instance: NS_A1_B2_B2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -378.1353760, 1136.2581787, -373.0659790, 1123.5406494, -1501.6760254, 1509.3239746
1: -291.8923645, 755.3835449, -288.1848450, 746.4497070, -1038.3419189, 1043.5682373
2: -315.8110352, 717.4314575, -311.8339233, 709.3007812, -1025.1116943, 1029.2653809
3: -317.3814087, 920.3523560, -313.5817566, 909.4652710, -1226.8466797, 1233.9340820
4: -460.3802185, 777.1884155, -454.9142456, 768.2630005, -1228.6431885, 1232.1026611

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3084622, upper bound: 1379.3043055
time: 0.80 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B1

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3073263, upper bound: 1379.3045800
time: 0.73 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2_B2_B2

### Relational analysis result of NS_A1_B2_B2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3066281, upper bound: 1379.3045644
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -321.8771362, 969.6198120, -330.2405090, 993.6204834, -1315.4975586, 1299.8603516
1: -248.3694458, 643.0812988, -254.7778778, 659.3219604, -907.6913452, 897.8591919
2: -268.8417664, 611.3493652, -275.6751404, 626.4904785, -895.3320923, 887.0245361
3: -270.6230164, 783.6445312, -277.3078003, 803.4201050, -1074.0429688, 1060.9523926
4: -392.3271790, 662.3067017, -402.2677612, 678.8594360, -1071.1861572, 1064.5744629

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3131531, upper bound: 1379.3097324
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156312, upper bound: 1379.3156667
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -370.2542419, 1116.0351562, -330.2405090, 993.6204834, -1363.8747559, 1446.2756348
1: -286.2065125, 741.5526733, -254.7778778, 659.3219604, -945.5284424, 996.3305664
2: -309.7839050, 704.7974854, -275.6751404, 626.4904785, -936.2744141, 980.4726562
3: -311.3895874, 903.3079834, -277.3078003, 803.4201050, -1114.8096924, 1180.6157227
4: -451.9976501, 763.4550781, -402.2677612, 678.8594360, -1130.8568115, 1165.7227783

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3100988, upper bound: 1379.3138329
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156312, upper bound: 1379.3168903
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -321.8771362, 969.6198120, -373.4217224, 1123.4768066, -1445.3540039, 1343.0415039
1: -248.3694458, 643.0812988, -288.5417175, 746.9279785, -995.2973633, 931.6230469
2: -268.8417664, 611.3493652, -312.2846069, 709.5980835, -978.4397583, 923.6339722
3: -270.6230164, 783.6445312, -313.6976013, 909.8775635, -1180.5006104, 1097.3421631
4: -392.3271790, 662.3067017, -455.3222961, 768.7992554, -1161.1259766, 1117.6290283

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3131531, upper bound: 1379.3097324
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3156500
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -370.2542419, 1116.0351562, -373.4217224, 1123.4768066, -1493.7310791, 1489.4569092
1: -286.2065125, 741.5526733, -288.5417175, 746.9279785, -1033.1342773, 1030.0943604
2: -309.7839050, 704.7974854, -312.2846069, 709.5980835, -1019.3819580, 1017.0820923
3: -311.3895874, 903.3079834, -313.6976013, 909.8775635, -1221.2670898, 1217.0056152
4: -451.9976501, 763.4550781, -455.3222961, 768.7992554, -1220.7966309, 1218.7773438

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3171966
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3150342, upper bound: 1379.3169699
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -321.8771362, 969.6198120, -336.7857361, 1012.2489014, -1334.1259766, 1306.4055176
1: -248.3694458, 643.0812988, -259.5390930, 671.5034180, -919.8728638, 902.6203613
2: -268.8417664, 611.3493652, -280.7144775, 637.8499756, -906.6916504, 892.0638428
3: -270.6230164, 783.6445312, -282.5847168, 818.5271606, -1089.1501465, 1066.2292480
4: -392.3271790, 662.3067017, -409.6542053, 691.1157227, -1083.4425049, 1071.9609375

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3068551, upper bound: 1379.3116870
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3113554, upper bound: 1379.3150511
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -370.2542419, 1116.0351562, -336.7857361, 1012.2489014, -1382.5031738, 1452.8209229
1: -286.2065125, 741.5526733, -259.5390930, 671.5034180, -957.7099609, 1001.0917969
2: -309.7839050, 704.7974854, -280.7144775, 637.8499756, -947.6339111, 985.5119019
3: -311.3895874, 903.3079834, -282.5847168, 818.5271606, -1129.9167480, 1185.8925781
4: -451.9976501, 763.4550781, -409.6542053, 691.1157227, -1143.1131592, 1173.1090088

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3103479, upper bound: 1379.3146160
time: 0.62 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3102118, upper bound: 1379.3082791
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -321.8771362, 969.6198120, -378.1353760, 1136.2581787, -1458.1352539, 1347.7551270
1: -248.3694458, 643.0812988, -291.8923645, 755.3835449, -1003.7529907, 934.9736328
2: -268.8417664, 611.3493652, -315.8110352, 717.4314575, -986.2731323, 927.1604004
3: -270.6230164, 783.6445312, -317.3814087, 920.3523560, -1190.9753418, 1101.0258789
4: -392.3271790, 662.3067017, -460.3802185, 777.1884155, -1169.5152588, 1122.6868896

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3046176, upper bound: 1379.3086425
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3046037, upper bound: 1379.3081297
time: 0.67 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -370.2542419, 1116.0351562, -378.1353760, 1136.2581787, -1506.5123291, 1494.1705322
1: -286.2065125, 741.5526733, -291.8923645, 755.3835449, -1041.5898438, 1033.4450684
2: -309.7839050, 704.7974854, -315.8110352, 717.4314575, -1027.2153320, 1020.6085205
3: -311.3895874, 903.3079834, -317.3814087, 920.3523560, -1231.7418213, 1220.6894531
4: -451.9976501, 763.4550781, -460.3802185, 777.1884155, -1229.1860352, 1223.8350830

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3046176, upper bound: 1379.3095742
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3046037, upper bound: 1379.3081297
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -325.6672668, 979.7856445, -330.2405090, 993.6204834, -1319.2877197, 1310.0261230
1: -251.0235138, 649.6229858, -254.7778778, 659.3219604, -910.3453979, 904.4008789
2: -271.6387329, 617.2684937, -275.6751404, 626.4904785, -898.1292114, 892.9436035
3: -273.6169739, 791.9397583, -277.3078003, 803.4201050, -1077.0371094, 1069.2471924
4: -396.4029541, 668.7114868, -402.2677612, 678.8594360, -1075.2622070, 1070.9792480

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3129919, upper bound: 1379.3086637
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3154650, upper bound: 1379.3126465
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -325.6672668, 979.7856445, -373.4217224, 1123.4768066, -1449.1440430, 1353.2073975
1: -251.0235138, 649.6229858, -288.5417175, 746.9279785, -997.9514771, 938.1646729
2: -271.6387329, 617.2684937, -312.2846069, 709.5980835, -981.2368164, 929.5530396
3: -273.6169739, 791.9397583, -313.6976013, 909.8775635, -1183.4945068, 1105.6372070
4: -396.4029541, 668.7114868, -455.3222961, 768.7992554, -1165.2021484, 1124.0338135

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3129919, upper bound: 1379.3086637
time: 0.76 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3154650, upper bound: 1379.3126466
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -373.0659790, 1123.5406494, -330.2405090, 993.6204834, -1366.6864014, 1453.7811279
1: -288.1848450, 746.4497070, -254.7778778, 659.3219604, -947.5065918, 1001.2276001
2: -311.8339233, 709.3007812, -275.6751404, 626.4904785, -938.3244019, 984.9759521
3: -313.5817566, 909.4652710, -277.3078003, 803.4201050, -1117.0018311, 1186.7730713
4: -454.9142456, 768.2630005, -402.2677612, 678.8594360, -1133.7735596, 1170.5307617

Time for backsubstitution: 1.99 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3100484, upper bound: 1379.3109339
time: 0.92 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -373.0659790, 1123.5406494, -373.4217224, 1123.4768066, -1496.5427246, 1496.9624023
1: -288.1848450, 746.4497070, -288.5417175, 746.9279785, -1035.1127930, 1034.9912109
2: -311.8339233, 709.3007812, -312.2846069, 709.5980835, -1021.4320068, 1021.5853882
3: -313.5817566, 909.4652710, -313.6976013, 909.8775635, -1223.4593506, 1223.1628418
4: -454.9142456, 768.2630005, -455.3222961, 768.7992554, -1223.7133789, 1223.5853271

Time for backsubstitution: 1.93 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3140860, upper bound: 1379.3102212
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
time: 0.65 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B1_A1

### Backsubstitution after applying NS history:
0: -325.6672668, 979.7856445, -336.7857361, 1012.2489014, -1337.9161377, 1316.5714111
1: -251.0235138, 649.6229858, -259.5390930, 671.5034180, -922.5269165, 909.1621094
2: -271.6387329, 617.2684937, -280.7144775, 637.8499756, -909.4887085, 897.9828491
3: -273.6169739, 791.9397583, -282.5847168, 818.5271606, -1092.1441650, 1074.5241699
4: -396.4029541, 668.7114868, -409.6542053, 691.1157227, -1087.5185547, 1078.3657227

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_B1_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112048, upper bound: 1379.3086324
time: 0.75 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A1_A2

### Relational analysis result of NS_A2_B1_A2_B2_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112575, upper bound: 1379.3109663
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B1_A2

### Backsubstitution after applying NS history:
0: -373.0659790, 1123.5406494, -336.7857361, 1012.2489014, -1385.3148193, 1460.3264160
1: -288.1848450, 746.4497070, -259.5390930, 671.5034180, -959.6881714, 1005.9887695
2: -311.8339233, 709.3007812, -280.7144775, 637.8499756, -949.6838989, 990.0151978
3: -313.5817566, 909.4652710, -282.5847168, 818.5271606, -1132.1088867, 1192.0499268
4: -454.9142456, 768.2630005, -409.6542053, 691.1157227, -1146.0297852, 1177.9172363

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_B1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3068165, upper bound: 1379.3102545
time: 1.27 seconds

## Relational analysis of NS_A2_B1_A2_B2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3112575, upper bound: 1379.3109663
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B2_A1

### Backsubstitution after applying NS history:
0: -325.6672668, 979.7856445, -378.1353760, 1136.2581787, -1461.9254150, 1357.9210205
1: -251.0235138, 649.6229858, -291.8923645, 755.3835449, -1006.4070435, 941.5153809
2: -271.6387329, 617.2684937, -315.8110352, 717.4314575, -989.0701904, 933.0795288
3: -273.6169739, 791.9397583, -317.3814087, 920.3523560, -1193.9693604, 1109.3210449
4: -396.4029541, 668.7114868, -460.3802185, 777.1884155, -1173.5913086, 1129.0916748

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 11

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 34

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 34

### Candidate
type: B, layer: 1, pos: 11

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_A1

### Relational analysis result of NS_A2_B1_A2_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -1379.3048301, upper bound: 1379.3036158
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2_A1_A2

### Relational analysis result of NS_A2_B1_A2_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3085494
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A2_B2_B2_A2

### Backsubstitution after applying NS history:
0: -373.0659790, 1123.5406494, -378.1353760, 1136.2581787, -1509.3239746, 1501.6760254
1: -288.1848450, 746.4497070, -291.8923645, 755.3835449, -1043.5682373, 1038.3419189
2: -311.8339233, 709.3007812, -315.8110352, 717.4314575, -1029.2652588, 1025.1118164
3: -313.5817566, 909.4652710, -317.3814087, 920.3523560, -1233.9340820, 1226.8466797
4: -454.9142456, 768.2630005, -460.3802185, 777.1884155, -1232.1026611, 1228.6431885

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A2_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3043055, upper bound: 1379.3085743
time: 1.07 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2_A2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 22

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A2_B1_A2_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B1_A2_B2_B2_A2_A1

### Relational analysis result of NS_A2_B1_A2_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3045800, upper bound: 1379.3073263
time: 0.63 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2_A2_A2

### Relational analysis result of NS_A2_B1_A2_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3045644, upper bound: 1379.3066281
time: 0.68 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -321.8771362, 969.6198120, -321.8771362, 969.6198120, -1291.4969482, 1291.4969482
1: -248.3694458, 643.0812988, -248.3694458, 643.0812988, -891.4507446, 891.4507446
2: -268.8417664, 611.3493652, -268.8417664, 611.3493652, -880.1910400, 880.1910400
3: -270.6230164, 783.6445312, -270.6230164, 783.6445312, -1054.2675781, 1054.2675781
4: -392.3271790, 662.3067017, -392.3271790, 662.3067017, -1054.6337891, 1054.6337891

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3097310, upper bound: 1379.3130824
time: 0.65 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156701, upper bound: 1379.3156550
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -321.8771362, 969.6198120, -370.2542419, 1116.0351562, -1437.9121094, 1339.8740234
1: -248.3694458, 643.0812988, -286.2065125, 741.5526733, -989.9221191, 929.2878418
2: -268.8417664, 611.3493652, -309.7839050, 704.7974854, -973.6391602, 921.1333008
3: -270.6230164, 783.6445312, -311.3895874, 903.3079834, -1173.9309082, 1095.0341797
4: -392.3271790, 662.3067017, -451.9976501, 763.4550781, -1155.7818604, 1114.3043213

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3130824, upper bound: 1379.3097322
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156701, upper bound: 1379.3156550
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -370.2542419, 1116.0351562, -321.8771362, 969.6198120, -1339.8740234, 1437.9121094
1: -286.2065125, 741.5526733, -248.3694458, 643.0812988, -929.2878418, 989.9221191
2: -309.7839050, 704.7974854, -268.8417664, 611.3493652, -921.1333008, 973.6391602
3: -311.3895874, 903.3079834, -270.6230164, 783.6445312, -1095.0341797, 1173.9309082
4: -451.9976501, 763.4550781, -392.3271790, 662.3067017, -1114.3043213, 1155.7818604

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3097322, upper bound: 1379.3137774
time: 0.62 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156702, upper bound: 1379.3169024
time: 0.77 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -370.2542419, 1116.0351562, -370.2542419, 1116.0351562, -1486.2891846, 1486.2891846
1: -286.2065125, 741.5526733, -286.2065125, 741.5526733, -1027.7591553, 1027.7590332
2: -309.7839050, 704.7974854, -309.7839050, 704.7974854, -1014.5814209, 1014.5814209
3: -311.3895874, 903.3079834, -311.3895874, 903.3079834, -1214.6973877, 1214.6973877
4: -451.9976501, 763.4550781, -451.9976501, 763.4550781, -1215.4525146, 1215.4525146

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156278, upper bound: 1379.3185932
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156702, upper bound: 1379.3176551
time: 0.67 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -321.8771362, 969.6198120, -325.6672668, 979.7856445, -1301.6628418, 1295.2871094
1: -248.3694458, 643.0812988, -251.0235138, 649.6229858, -897.9924316, 894.1047974
2: -268.8417664, 611.3493652, -271.6387329, 617.2684937, -886.1101074, 882.9880981
3: -270.6230164, 783.6445312, -273.6169739, 791.9397583, -1062.5625000, 1057.2614746
4: -392.3271790, 662.3067017, -396.4029541, 668.7114868, -1061.0384521, 1058.7097168

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3086407, upper bound: 1379.3125921
time: 0.67 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3126401, upper bound: 1379.3154058
time: 0.74 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -370.2542419, 1116.0351562, -325.6672668, 979.7856445, -1350.0399170, 1441.7023926
1: -286.2065125, 741.5526733, -251.0235138, 649.6229858, -935.8294678, 992.5761719
2: -309.7839050, 704.7974854, -271.6387329, 617.2684937, -927.0523682, 976.4362183
3: -311.3895874, 903.3079834, -273.6169739, 791.9397583, -1103.3291016, 1176.9249268
4: -451.9976501, 763.4550781, -396.4029541, 668.7114868, -1120.7091064, 1159.8580322

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3103469, upper bound: 1379.3145762
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A1_B2_B1_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3102107, upper bound: 1379.3082734
time: 0.75 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -321.8771362, 969.6198120, -373.0659790, 1123.5406494, -1445.4177246, 1342.6857910
1: -248.3694458, 643.0812988, -288.1848450, 746.4497070, -994.8191528, 931.2660522
2: -268.8417664, 611.3493652, -311.8339233, 709.3007812, -978.1424561, 923.1832886
3: -270.6230164, 783.6445312, -313.5817566, 909.4652710, -1180.0882568, 1097.2263184
4: -392.3271790, 662.3067017, -454.9142456, 768.2630005, -1160.5902100, 1117.2209473

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3098375, upper bound: 1379.3086572
time: 0.71 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3139227
time: 0.71 seconds

## BFS NS instance: NS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -370.2542419, 1116.0351562, -373.0659790, 1123.5406494, -1493.7949219, 1489.1008301
1: -286.2065125, 741.5526733, -288.1848450, 746.4497070, -1032.6558838, 1029.7375488
2: -309.7839050, 704.7974854, -311.8339233, 709.3007812, -1019.0847168, 1016.6314087
3: -311.3895874, 903.3079834, -313.5817566, 909.4652710, -1220.8547363, 1216.8896484
4: -451.9976501, 763.4550781, -454.9142456, 768.2630005, -1220.2606201, 1218.3692627

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3067968, upper bound: 1379.3125840
time: 0.81 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A1_B2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3066799, upper bound: 1379.3081297
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -325.6672668, 979.7856445, -321.8771362, 969.6198120, -1295.2871094, 1301.6628418
1: -251.0235138, 649.6229858, -248.3694458, 643.0812988, -894.1047974, 897.9924316
2: -271.6387329, 617.2684937, -268.8417664, 611.3493652, -882.9880981, 886.1101074
3: -273.6169739, 791.9397583, -270.6230164, 783.6445312, -1057.2614746, 1062.5625000
4: -396.4029541, 668.7114868, -392.3271790, 662.3067017, -1058.7097168, 1061.0384521

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3129218, upper bound: 1379.3086623
time: 0.74 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3156200, upper bound: 1379.3126542
time: 0.60 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -325.6672668, 979.7856445, -370.2542419, 1116.0351562, -1441.7023926, 1350.0399170
1: -251.0235138, 649.6229858, -286.2065125, 741.5526733, -992.5761719, 935.8294678
2: -271.6387329, 617.2684937, -309.7839050, 704.7974854, -976.4362183, 927.0523682
3: -273.6169739, 791.9397583, -311.3895874, 903.3079834, -1176.9249268, 1103.3291016
4: -396.4029541, 668.7114868, -451.9976501, 763.4550781, -1159.8580322, 1120.7091064

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3145451, upper bound: 1379.3103450
time: 0.82 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3082734, upper bound: 1379.3102107
time: 0.70 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -373.0659790, 1123.5406494, -321.8771362, 969.6198120, -1342.6857910, 1445.4177246
1: -288.1848450, 746.4497070, -248.3694458, 643.0812988, -931.2660522, 994.8191528
2: -311.8339233, 709.3007812, -268.8417664, 611.3493652, -923.1832886, 978.1424561
3: -313.5817566, 909.4652710, -270.6230164, 783.6445312, -1097.2263184, 1180.0882568
4: -454.9142456, 768.2630005, -392.3271790, 662.3067017, -1117.2209473, 1160.5902100

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 44

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3096830, upper bound: 1379.3109282
time: 1.02 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3148870, upper bound: 1379.3110356
time: 0.72 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -373.0659790, 1123.5406494, -370.2542419, 1116.0351562, -1489.1008301, 1493.7949219
1: -288.1848450, 746.4497070, -286.2065125, 741.5526733, -1029.7375488, 1032.6558838
2: -311.8339233, 709.3007812, -309.7839050, 704.7974854, -1016.6314087, 1019.0847168
3: -313.5817566, 909.4652710, -311.3895874, 903.3079834, -1216.8896484, 1220.8547363
4: -454.9142456, 768.2630005, -451.9976501, 763.4550781, -1218.3692627, 1220.2606201

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3117294, upper bound: 1379.3068161
time: 0.83 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3081807, upper bound: 1379.3067222
time: 0.85 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -325.6672668, 979.7856445, -325.6672668, 979.7856445, -1305.4528809, 1305.4528809
1: -251.0235138, 649.6229858, -251.0235138, 649.6229858, -900.6464844, 900.6464844
2: -271.6387329, 617.2684937, -271.6387329, 617.2684937, -888.9072266, 888.9072266
3: -273.6169739, 791.9397583, -273.6169739, 791.9397583, -1065.5566406, 1065.5566406
4: -396.4029541, 668.7114868, -396.4029541, 668.7114868, -1065.1145020, 1065.1145020

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3086290, upper bound: 1379.3114744
time: 0.63 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3120980
time: 0.69 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -325.6672668, 979.7856445, -373.0659790, 1123.5406494, -1449.2078857, 1352.8515625
1: -251.0235138, 649.6229858, -288.1848450, 746.4497070, -997.4732056, 937.8077393
2: -271.6387329, 617.2684937, -311.8339233, 709.3007812, -980.9395142, 929.1024170
3: -273.6169739, 791.9397583, -313.5817566, 909.4652710, -1183.0822754, 1105.5212402
4: -396.4029541, 668.7114868, -454.9142456, 768.2630005, -1164.6660156, 1123.6257324

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3104664, upper bound: 1379.3085599
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3120980
time: 0.82 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -373.0659790, 1123.5406494, -325.6672668, 979.7856445, -1352.8515625, 1449.2078857
1: -288.1848450, 746.4497070, -251.0235138, 649.6229858, -937.8077393, 997.4732056
2: -311.8339233, 709.3007812, -271.6387329, 617.2684937, -929.1024170, 980.9395142
3: -313.5817566, 909.4652710, -273.6169739, 791.9397583, -1105.5212402, 1183.0822754
4: -454.9142456, 768.2630005, -396.4029541, 668.7114868, -1123.6257324, 1164.6660156

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3086125, upper bound: 1379.3108902
time: 0.77 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
time: 0.64 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -373.0659790, 1123.5406494, -373.0659790, 1123.5406494, -1496.6066895, 1496.6066895
1: -288.1848450, 746.4497070, -288.1848450, 746.4497070, -1034.6343994, 1034.6342773
2: -311.8339233, 709.3007812, -311.8339233, 709.3007812, -1021.1347046, 1021.1347046
3: -313.5817566, 909.4652710, -313.5817566, 909.4652710, -1223.0468750, 1223.0468750
4: -454.9142456, 768.2630005, -454.9142456, 768.2630005, -1223.1772461, 1223.1772461

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 40

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 24

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3086125, upper bound: 1379.3108902
time: 0.72 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
time: 0.67 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.40 seconds
NS_A1_B1_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3132005, upper bound: 1379.3101000
NS_A1_B1_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156111, upper bound: 1379.3155131
NS_A1_B1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3101000, upper bound: 1379.3134649
NS_A1_B1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156111, upper bound: 1379.3155131
NS_A1_B1_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3132005, upper bound: 1379.3101000
NS_A1_B1_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154949
NS_A1_B1_A1_B1_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3132005, upper bound: 1379.3123640
NS_A1_B1_A1_B1_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3154950, upper bound: 1379.3154950
NS_A1_B1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3068580, upper bound: 1379.3117708
NS_A1_B1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3068580, upper bound: 1379.3150697
NS_A1_B1_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3068580, upper bound: 1379.3121291
NS_A1_B1_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3113394, upper bound: 1379.3150697
NS_A1_B1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3046254, upper bound: 1379.3088612
NS_A1_B1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3046160, upper bound: 1379.3085028
NS_A1_B1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3046253, upper bound: 1379.3093303
NS_A1_B1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3046159, upper bound: 1379.3085028
NS_A1_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3117708, upper bound: 1379.3068580
NS_A1_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3150697, upper bound: 1379.3113394
NS_A1_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3117708, upper bound: 1379.3068584
NS_A1_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3150697, upper bound: 1379.3113394
NS_A1_B2_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3097324, upper bound: 1379.3131531
NS_A1_B2_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156667, upper bound: 1379.3156312
NS_A1_B2_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3131281, upper bound: 1379.3100999
NS_A1_B2_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156667, upper bound: 1379.3156312
NS_A1_B2_B1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3097324, upper bound: 1379.3134207
NS_A1_B2_B1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
NS_A1_B2_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3155139
NS_A1_B2_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156500, upper bound: 1379.3150342
NS_A1_B2_B1_A2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3116870, upper bound: 1379.3068551
NS_A1_B2_B1_A2_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3150511, upper bound: 1379.3113554
NS_A1_B2_B1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3146160, upper bound: 1379.3103479
NS_A1_B2_B1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3082791, upper bound: 1379.3102118
NS_A1_B2_B1_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3086425, upper bound: 1379.3046175
NS_A1_B2_B1_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3081297, upper bound: 1379.3046037
NS_A1_B2_B1_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3086425, upper bound: 1379.3046176
NS_A1_B2_B1_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3081297, upper bound: 1379.3046037
NS_A1_B2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3086637, upper bound: 1379.3129919
NS_A1_B2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3126465, upper bound: 1379.3154650
NS_A1_B2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3086637, upper bound: 1379.3132686
NS_A1_B2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3126466, upper bound: 1379.3154652
NS_A1_B2_B2_A1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3109339, upper bound: 1379.3100484
NS_A1_B2_B2_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3149152
NS_A1_B2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3086637, upper bound: 1379.3140796
NS_A1_B2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3110343, upper bound: 1379.3150299
NS_A1_B2_B2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3086323, upper bound: 1379.3112048
NS_A1_B2_B2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3109663, upper bound: 1379.3112575
NS_A1_B2_B2_A2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3102545, upper bound: 1379.3068165
NS_A1_B2_B2_A2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3109663, upper bound: 1379.3112575
NS_A1_B2_B2_A2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3036158, upper bound: 1379.3048301
NS_A1_B2_B2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3085494, upper bound: 1379.3050145
NS_A1_B2_B2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3073263, upper bound: 1379.3045800
NS_A1_B2_B2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3066281, upper bound: 1379.3045644
NS_A2_B1_A1_B1_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3131531, upper bound: 1379.3097324
NS_A2_B1_A1_B1_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156312, upper bound: 1379.3156667
NS_A2_B1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3100988, upper bound: 1379.3138329
NS_A2_B1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156312, upper bound: 1379.3168903
NS_A2_B1_A1_B1_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3131531, upper bound: 1379.3097324
NS_A2_B1_A1_B1_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3156500
NS_A2_B1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3155139, upper bound: 1379.3171966
NS_A2_B1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3150342, upper bound: 1379.3169699
NS_A2_B1_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3068551, upper bound: 1379.3116870
NS_A2_B1_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3113554, upper bound: 1379.3150511
NS_A2_B1_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3103479, upper bound: 1379.3146160
NS_A2_B1_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3102118, upper bound: 1379.3082791
NS_A2_B1_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3046176, upper bound: 1379.3086425
NS_A2_B1_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3046037, upper bound: 1379.3081297
NS_A2_B1_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3046176, upper bound: 1379.3095742
NS_A2_B1_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3046037, upper bound: 1379.3081297
NS_A2_B1_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3129919, upper bound: 1379.3086637
NS_A2_B1_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3154650, upper bound: 1379.3126465
NS_A2_B1_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3129919, upper bound: 1379.3086637
NS_A2_B1_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3154650, upper bound: 1379.3126466
NS_A2_B1_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3100484, upper bound: 1379.3109339
NS_A2_B1_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
NS_A2_B1_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3140860, upper bound: 1379.3102212
NS_A2_B1_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3149152, upper bound: 1379.3110343
NS_A2_B1_A2_B2_B1_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3112048, upper bound: 1379.3086324
NS_A2_B1_A2_B2_B1_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3112575, upper bound: 1379.3109663
NS_A2_B1_A2_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3068165, upper bound: 1379.3102545
NS_A2_B1_A2_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3112575, upper bound: 1379.3109663
NS_A2_B1_A2_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3048301, upper bound: 1379.3036158
NS_A2_B1_A2_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3050145, upper bound: 1379.3085494
NS_A2_B1_A2_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3045800, upper bound: 1379.3073263
NS_A2_B1_A2_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3045644, upper bound: 1379.3066281
NS_A2_B2_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3097310, upper bound: 1379.3130824
NS_A2_B2_A1_B1_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156701, upper bound: 1379.3156550
NS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3130824, upper bound: 1379.3097322
NS_A2_B2_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156701, upper bound: 1379.3156550
NS_A2_B2_A1_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3097322, upper bound: 1379.3137774
NS_A2_B2_A1_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156702, upper bound: 1379.3169024
NS_A2_B2_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156278, upper bound: 1379.3185932
NS_A2_B2_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156702, upper bound: 1379.3176551
NS_A2_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3086407, upper bound: 1379.3125921
NS_A2_B2_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3126401, upper bound: 1379.3154058
NS_A2_B2_A1_B2_B1_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3103469, upper bound: 1379.3145762
NS_A2_B2_A1_B2_B1_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3102107, upper bound: 1379.3082734
NS_A2_B2_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3098375, upper bound: 1379.3086572
NS_A2_B2_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3110065, upper bound: 1379.3139227
NS_A2_B2_A1_B2_B2_A2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3067968, upper bound: 1379.3125840
NS_A2_B2_A1_B2_B2_A2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3066799, upper bound: 1379.3081297
NS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3129218, upper bound: 1379.3086623
NS_A2_B2_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3156200, upper bound: 1379.3126542
NS_A2_B2_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3145451, upper bound: 1379.3103450
NS_A2_B2_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3082734, upper bound: 1379.3102107
NS_A2_B2_A2_B1_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3096830, upper bound: 1379.3109282
NS_A2_B2_A2_B1_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3148870, upper bound: 1379.3110356
NS_A2_B2_A2_B1_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3117294, upper bound: 1379.3068161
NS_A2_B2_A2_B1_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3081807, upper bound: 1379.3067222
NS_A2_B2_A2_B2_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3086290, upper bound: 1379.3114744
NS_A2_B2_A2_B2_A1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3120980
NS_A2_B2_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3104664, upper bound: 1379.3085599
NS_A2_B2_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3120980
NS_A2_B2_A2_B2_A2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3086125, upper bound: 1379.3108902
NS_A2_B2_A2_B2_A2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514
NS_A2_B2_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3086125, upper bound: 1379.3108902
NS_A2_B2_A2_B2_A2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.40
Output dim: 0, lower bound: -1379.3109514, upper bound: 1379.3109514

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.49 + 417.54 = 421.03 seconds
