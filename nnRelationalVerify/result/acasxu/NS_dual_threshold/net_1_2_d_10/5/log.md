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
execution time: IAR + RelationalAnalysis = 1.56 + 1.90 = 3.45 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -817.2512044, upper bound: 817.2512044

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 44
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 44

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2511253, upper bound: 817.2506496
time: 0.66 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2506928, upper bound: 817.2506928
time: 0.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.49 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -817.2511253, upper bound: 817.2506496
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.49
Output dim: 0, lower bound: -817.2506928, upper bound: 817.2506928

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -173.2468872, 696.0963745, -180.8047333, 726.3649902, -899.6118774, 876.9011230
1: -214.3323364, 786.8086548, -223.6785431, 820.9863892, -1035.3187256, 1010.4871826
2: -224.8905182, 797.4176025, -234.7946930, 832.1535034, -1057.0440674, 1032.2122803
3: -356.7940063, 842.0077515, -372.3912354, 878.7119141, -1235.5058594, 1214.3989258
4: -362.7246704, 810.1069946, -378.6652832, 845.5180054, -1208.2426758, 1188.7720947

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_A1

### Relational analysis result of NS_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2432803, upper bound: 817.2382939
time: 0.80 seconds

## Relational analysis of NS_A1_A2

### Relational analysis result of NS_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2511253, upper bound: 817.2506496
time: 0.69 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -182.4034271, 732.1229248, -185.6558533, 745.0399170, -927.4432373, 917.7788086
1: -225.7174683, 827.4150391, -229.7426910, 842.1028442, -1067.8203125, 1057.1577148
2: -236.8057861, 838.8110962, -241.0974274, 853.6782227, -1090.4840088, 1079.9085693
3: -375.5668335, 885.5974121, -382.3359985, 901.4897461, -1277.0566406, 1267.9332275
4: -381.7607727, 852.1938477, -388.7119446, 867.4756470, -1249.2364502, 1240.9056396

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2488899, upper bound: 817.2497150
time: 1.32 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2488802, upper bound: 817.2489432
time: 0.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.60 seconds
NS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -817.2432803, upper bound: 817.2382939
NS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -817.2511253, upper bound: 817.2506496
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -817.2488899, upper bound: 817.2497150
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.60
Output dim: 0, lower bound: -817.2488802, upper bound: 817.2489432

## BFS NS instance: NS_A1_A1

### Backsubstitution after applying NS history:
0: -167.4388123, 670.8915405, -176.1551971, 707.6030273, -875.0418091, 847.0466919
1: -207.1676636, 758.2422485, -217.8992462, 799.7814941, -1006.9490967, 976.1414795
2: -217.2238312, 768.6457520, -228.7401276, 810.6215820, -1027.8454590, 997.3858643
3: -344.3521729, 811.5833130, -362.7194519, 856.0261230, -1200.3782959, 1174.3027344
4: -350.1717529, 780.6815186, -368.9226990, 823.5379028, -1173.7094727, 1149.6041260

Time for backsubstitution: 1.44 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2419255, upper bound: 817.2341786
time: 0.69 seconds

## Relational analysis of NS_A1_A1_A2

### Relational analysis result of NS_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2223537, upper bound: 817.2313790
time: 1.13 seconds

## BFS NS instance: NS_A1_A2

### Backsubstitution after applying NS history:
0: -168.4340363, 676.5480957, -178.6127167, 717.4961548, -885.9301147, 855.1607666
1: -208.4537659, 764.8114624, -220.9957275, 811.0053101, -1019.4591064, 985.8071899
2: -218.7468567, 775.0723267, -231.9976349, 822.0164795, -1040.7633057, 1007.0699463
3: -346.9899902, 818.6329956, -367.9211731, 868.1016846, -1215.0916748, 1186.5541992
4: -352.8601990, 787.4381104, -374.1790466, 835.2314453, -1188.0914307, 1161.6166992

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_A1

### Relational analysis result of NS_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501235, upper bound: 817.2466265
time: 0.65 seconds

## Relational analysis of NS_A1_A2_A2

### Relational analysis result of NS_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2450127, upper bound: 817.2456411
time: 0.72 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -179.7738953, 721.5812378, -175.3437958, 703.8980713, -883.6719360, 896.9250488
1: -222.4507141, 815.4843750, -216.9217377, 795.5298462, -1017.9804688, 1032.4061279
2: -233.3754425, 826.6885376, -227.6485596, 806.3209839, -1039.6964111, 1054.3369141
3: -370.1051941, 872.7730713, -360.9512939, 851.3924561, -1221.4976807, 1233.7243652
4: -376.2058716, 839.8406372, -366.9273987, 819.2095337, -1195.4154053, 1206.7680664

Time for backsubstitution: 1.46 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 14

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2488802, upper bound: 817.2488802
time: 0.66 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2488802, upper bound: 817.2489432
time: 0.67 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -172.6898346, 691.9714966, -239.0418701, 970.0054321, -1140.3764648, 931.0133667
1: -213.6893463, 781.9545288, -295.6646729, 1095.6312256, -1306.4626465, 1077.6191406
2: -223.9772949, 792.8153076, -310.1849976, 1110.4472656, -1331.8238525, 1103.0001221
3: -355.3595886, 837.1618042, -495.2873840, 1173.4866943, -1526.3632812, 1332.4492188
4: -361.1712341, 805.5077515, -504.2026062, 1127.5017090, -1487.0040283, 1309.7103271

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2154330, upper bound: 817.2228572
time: 0.63 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2143340, upper bound: 817.2143340
time: 0.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 2.93 seconds
NS_A1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -817.2419255, upper bound: 817.2341786
NS_A1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -817.2223537, upper bound: 817.2313790
NS_A1_A2_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -817.2501235, upper bound: 817.2466265
NS_A1_A2_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -817.2450127, upper bound: 817.2456411
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -817.2488802, upper bound: 817.2488802
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 2.93
Output dim: 0, lower bound: -817.2488802, upper bound: 817.2489432
NS_A2_B2_B1, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 0, lower bound: -817.2154330, upper bound: 817.2228572
NS_A2_B2_B2, status: Status.VERIFIED, split count: 3, time: 2.93
Output dim: 0, lower bound: -817.2143340, upper bound: 817.2143340

## BFS NS instance: NS_A1_A1_A1

### Backsubstitution after applying NS history:
0: -160.0674896, 640.2591553, -174.3560181, 700.2781982, -860.3457031, 814.6151733
1: -198.0782623, 723.6499023, -215.6568756, 791.4837036, -989.5619507, 939.3067627
2: -207.6085510, 733.6979370, -226.3891754, 802.2306519, -1009.8392334, 960.0870972
3: -328.9580383, 774.5792236, -358.9521484, 847.1026001, -1176.0606689, 1133.5313721
4: -334.4498291, 745.2206421, -365.0860291, 814.9857788, -1149.4354248, 1110.3066406

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_A1_A1

### Relational analysis result of NS_A1_A1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2419255, upper bound: 817.2341786
time: 0.73 seconds

## Relational analysis of NS_A1_A1_A1_A2

### Relational analysis result of NS_A1_A1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2376484, upper bound: 817.2338745
time: 0.61 seconds

## BFS NS instance: NS_A1_A1_A2

### Backsubstitution after applying NS history:
0: -163.0966492, 653.2735596, -174.6434326, 701.5113525, -864.6080322, 827.9168701
1: -201.8013763, 738.2733154, -216.0256042, 792.8753052, -994.6766968, 954.2989502
2: -211.5215912, 748.4522095, -226.7523193, 803.6254883, -1015.1470337, 975.2044678
3: -335.3054504, 790.0597534, -359.5558777, 848.5590820, -1183.8645020, 1149.6156006
4: -340.8943176, 759.9857178, -365.6813965, 816.3513184, -1157.2454834, 1125.6668701

Time for backsubstitution: 1.45 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_A2_A1

### Relational analysis result of NS_A1_A1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2313737
time: 0.77 seconds

## Relational analysis of NS_A1_A1_A2_A2

### Relational analysis result of NS_A1_A1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2216185, upper bound: 817.2313790
time: 0.82 seconds

## BFS NS instance: NS_A1_A2_A1

### Backsubstitution after applying NS history:
0: -161.4837952, 647.7257080, -176.8017426, 710.1138306, -871.5975342, 824.5274048
1: -199.8806915, 732.2522583, -218.7395172, 802.6414185, -1002.5220947, 950.9917603
2: -209.6508942, 742.1520386, -229.6290436, 813.5598755, -1023.2107544, 971.7810669
3: -332.4407959, 783.7374878, -364.1273804, 859.1065674, -1191.5473633, 1147.8647461
4: -337.9813843, 753.9857788, -370.3154602, 826.6105957, -1164.5919189, 1124.3012695

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A1_A1

### Relational analysis result of NS_A1_A2_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2501235, upper bound: 817.2465135
time: 0.62 seconds

## Relational analysis of NS_A1_A2_A1_A2

### Relational analysis result of NS_A1_A2_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2499966, upper bound: 817.2466265
time: 0.68 seconds

## BFS NS instance: NS_A1_A2_A2

### Backsubstitution after applying NS history:
0: -164.1233521, 659.0985718, -177.1039734, 711.4216309, -875.5449829, 836.2024536
1: -203.1183319, 745.0406494, -219.1268311, 804.1198730, -1007.2381592, 964.1674805
2: -213.0722961, 755.0513916, -230.0140839, 815.0407715, -1028.1130371, 985.0654297
3: -337.9881287, 797.2699585, -364.7669983, 860.6559448, -1198.6440430, 1162.0364990
4: -343.6055298, 766.8981323, -370.9434509, 828.0682373, -1171.6735840, 1137.8415527

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A2_A1

### Relational analysis result of NS_A1_A2_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2450127, upper bound: 817.2456411
time: 0.78 seconds

## Relational analysis of NS_A1_A2_A2_A2

### Relational analysis result of NS_A1_A2_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2445993, upper bound: 817.2453182
time: 0.72 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -172.1723175, 691.3016968, -175.3437958, 703.8980713, -876.0703125, 866.6455078
1: -212.9961090, 781.2151489, -216.9217377, 795.5298462, -1008.5258789, 998.1368408
2: -223.4642487, 791.8212280, -227.6485596, 806.3209839, -1029.7849121, 1019.4697876
3: -354.3551025, 835.8966064, -360.9512939, 851.3924561, -1205.7475586, 1196.8479004
4: -360.1543579, 804.3106689, -366.9273987, 819.2095337, -1179.3638916, 1171.2377930

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2483197, upper bound: 817.2492205
time: 0.60 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2483197, upper bound: 817.2497150
time: 1.15 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -235.7100830, 956.7470703, -175.3437958, 703.8980713, -939.6081543, 1129.7327881
1: -291.5221863, 1080.5463867, -216.9217377, 795.5298462, -1087.0518799, 1294.5694580
2: -305.7863159, 1095.1661377, -227.6485596, 806.3209839, -1112.1072998, 1320.1925049
3: -488.3197632, 1157.1617432, -360.9512939, 851.3924561, -1339.7121582, 1515.5739746
4: -497.0578918, 1111.8228760, -366.9273987, 819.2095337, -1316.2674561, 1477.0791016

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A2_A1

### Relational analysis result of NS_A2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2259929, upper bound: 817.2224278
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A2_A2

### Relational analysis result of NS_A2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2149414, upper bound: 817.2207715
time: 0.88 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.22 seconds
NS_A1_A1_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2419255, upper bound: 817.2341786
NS_A1_A1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2376484, upper bound: 817.2338745
NS_A1_A1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2313737
NS_A1_A1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2216185, upper bound: 817.2313790
NS_A1_A2_A1_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2501235, upper bound: 817.2465135
NS_A1_A2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2499966, upper bound: 817.2466265
NS_A1_A2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2450127, upper bound: 817.2456411
NS_A1_A2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2445993, upper bound: 817.2453182
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2483197, upper bound: 817.2492205
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2483197, upper bound: 817.2497150
NS_A2_B1_A2_A1, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2259929, upper bound: 817.2224278
NS_A2_B1_A2_A2, status: Status.VERIFIED, split count: 4, time: 3.22
Output dim: 0, lower bound: -817.2149414, upper bound: 817.2207715

## BFS NS instance: NS_A1_A1_A1_A1

### Backsubstitution after applying NS history:
0: -139.8680878, 557.6105957, -166.4869995, 669.4533081, -809.3213501, 724.0975952
1: -173.1533661, 630.2206421, -205.9283295, 756.6007080, -929.7540283, 836.1489258
2: -181.3994293, 639.3145142, -216.1799622, 766.8004150, -948.1998291, 855.4944458
3: -286.9536743, 674.6697388, -342.7958984, 809.5985107, -1096.5522461, 1017.4656372
4: -291.6203003, 649.4708862, -348.6484985, 778.8814087, -1070.5017090, 998.1193848

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 0

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A1_A1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2419255, upper bound: 817.2341786
time: 0.65 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2419255, upper bound: 817.2341786
time: 0.81 seconds

## BFS NS instance: NS_A1_A1_A1_A2

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -170.9287415, 686.8590698, -842.1672363, 792.8499756
1: -192.1896210, 702.8735962, -211.4213257, 776.2759399, -968.4655151, 914.2947998
2: -201.3675842, 712.5927124, -221.8842773, 786.8109131, -988.1784668, 934.4769287
3: -319.2729492, 752.2887573, -351.9332886, 830.7967529, -1150.0697021, 1104.2218018
4: -324.6016541, 723.6882324, -357.9486389, 799.2484131, -1123.8500977, 1081.6365967

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_A1_A2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2272484
time: 0.91 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2338745
time: 0.73 seconds

## BFS NS instance: NS_A1_A1_A2_A1

### Backsubstitution after applying NS history:
0: -143.8092804, 574.0617676, -166.7565460, 670.5975342, -814.4067383, 740.8182983
1: -178.0216064, 648.7533569, -206.2770386, 757.8884888, -935.9100952, 855.0303955
2: -186.5074158, 658.0904541, -216.5173645, 768.0964966, -954.6038818, 874.6076660
3: -295.1674500, 694.4026489, -343.3612366, 810.9431763, -1106.1105957, 1037.7639160
4: -299.9495239, 668.4326782, -349.1981812, 780.1490479, -1080.0983887, 1017.6308594

Time for backsubstitution: 1.47 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A2_A1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2313737
time: 0.64 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2313737
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A2_A2

### Backsubstitution after applying NS history:
0: -158.3521729, 634.9489136, -171.2529602, 688.2500000, -846.6021118, 806.2017822
1: -195.9344635, 717.5258179, -211.8352203, 777.8474731, -973.7819214, 929.3610229
2: -205.3063965, 727.3727417, -222.2976532, 788.3847656, -993.6911621, 949.6704102
3: -325.6563110, 767.8197021, -352.6185913, 832.4481201, -1158.1043701, 1120.4382324
4: -331.0854797, 738.4993286, -358.6263428, 800.7977905, -1131.8833008, 1097.1253662

Time for backsubstitution: 1.48 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A1_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_A2_A2_B1

### Relational analysis result of NS_A1_A1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2210498, upper bound: 817.2237314
time: 1.18 seconds

## Relational analysis of NS_A1_A1_A2_A2_B2

### Relational analysis result of NS_A1_A1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2210498, upper bound: 817.2313790
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_A1_A1

### Backsubstitution after applying NS history:
0: -142.6351776, 571.1234741, -169.0159454, 679.6384277, -822.2736206, 740.1394043
1: -176.5977631, 645.6719360, -209.1058044, 768.1486206, -944.7463379, 854.7777100
2: -185.1582794, 654.6025391, -219.5174866, 778.5169678, -963.6752319, 874.1199951
3: -293.1644592, 690.7665405, -348.1227112, 822.0138550, -1115.1783447, 1038.8890381
4: -297.7338257, 665.0226440, -354.0546265, 790.8723145, -1088.6062012, 1019.0772095

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_A1_A1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2447706
time: 0.77 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2465135
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_A1_A2

### Backsubstitution after applying NS history:
0: -156.2181702, 627.3008423, -173.3845062, 696.7471924, -852.9653320, 800.6853638
1: -193.3670197, 709.1093140, -214.5136108, 787.4957886, -980.8627319, 923.6229248
2: -202.7233887, 718.6606445, -225.1405640, 798.1999512, -1000.9233398, 943.8012085
3: -321.6897888, 758.8978882, -357.1290588, 842.8679810, -1164.5577393, 1116.0266113
4: -327.0349426, 729.9923096, -363.2036133, 810.9323120, -1137.9672852, 1093.1959229

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_A1_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A1_A2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2497985, upper bound: 817.2454987
time: 0.70 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2497985, upper bound: 817.2466265
time: 0.88 seconds

## BFS NS instance: NS_A1_A2_A2_A1

### Backsubstitution after applying NS history:
0: -145.9492340, 584.7414551, -169.2966156, 680.8276367, -826.7768555, 754.0380859
1: -180.7044830, 661.0221558, -209.4684906, 769.4879761, -950.1923828, 870.4906616
2: -189.4848633, 670.1931763, -219.8721161, 779.8659058, -969.3507690, 890.0653076
3: -300.1191101, 707.2645264, -348.7145691, 823.4163208, -1123.5354004, 1055.9790039
4: -304.8388977, 680.8472290, -354.6297302, 792.1985474, -1097.0371094, 1035.4768066

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_A2_A1_B1

### Relational analysis result of NS_A1_A2_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2444322
time: 0.84 seconds

## Relational analysis of NS_A1_A2_A2_A1_B2

### Relational analysis result of NS_A1_A2_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2456411
time: 0.80 seconds

## BFS NS instance: NS_A1_A2_A2_A2

### Backsubstitution after applying NS history:
0: -158.8125610, 638.4288330, -173.7238312, 698.2064819, -857.0189819, 812.1526489
1: -196.5520020, 721.6295166, -214.9468689, 789.1463013, -985.6983032, 936.5764160
2: -206.0884094, 731.2836914, -225.5758362, 799.8544312, -1005.9428711, 956.8594971
3: -327.1409607, 772.1646118, -357.8498840, 844.6055908, -1171.7464600, 1130.0142822
4: -332.5650330, 742.6354980, -363.9148254, 812.5694580, -1145.1345215, 1106.5502930

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_A2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A2_A2_B1

### Relational analysis result of NS_A1_A2_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2445322, upper bound: 817.2449159
time: 0.75 seconds

## Relational analysis of NS_A1_A2_A2_A2_B2

### Relational analysis result of NS_A1_A2_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2445322, upper bound: 817.2453182
time: 0.71 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -172.1723175, 691.3016968, -163.5871429, 657.5558472, -829.7281494, 854.8888550
1: -212.9961090, 781.2151489, -202.3060150, 743.2506104, -956.2466431, 983.5209961
2: -223.4642487, 791.8212280, -212.2849884, 753.0233765, -976.4875488, 1004.1062012
3: -354.3551025, 835.8966064, -336.7553101, 795.1429443, -1149.4978027, 1172.6517334
4: -360.1543579, 804.3106689, -342.3205261, 764.8621216, -1125.0164795, 1146.6309814

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2484142, upper bound: 817.2452156
time: 0.70 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2456384, upper bound: 817.2450553
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -172.1723175, 691.3016968, -172.1723175, 691.3016968, -863.4739990, 863.4739990
1: -212.9961090, 781.2151489, -212.9961090, 781.2151489, -994.2111816, 994.2111816
2: -223.4642487, 791.8212280, -223.4642487, 791.8212280, -1015.2854004, 1015.2854004
3: -354.3551025, 835.8966064, -354.3551025, 835.8966064, -1190.2515869, 1190.2515869
4: -360.1543579, 804.3106689, -360.1543579, 804.3106689, -1164.4649658, 1164.4648438

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2466165, upper bound: 817.2498619
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2456384, upper bound: 817.2467862
time: 0.69 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.04 seconds
NS_A1_A1_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2419255, upper bound: 817.2341786
NS_A1_A1_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2419255, upper bound: 817.2341786
NS_A1_A1_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2272484
NS_A1_A1_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2338745
NS_A1_A1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2313737
NS_A1_A1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2313737
NS_A1_A1_A2_A2_B1, status: Status.VERIFIED, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2210498, upper bound: 817.2237314
NS_A1_A1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2210498, upper bound: 817.2313790
NS_A1_A2_A1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2447706
NS_A1_A2_A1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2465135
NS_A1_A2_A1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2497985, upper bound: 817.2454987
NS_A1_A2_A1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2497985, upper bound: 817.2466265
NS_A1_A2_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2444322
NS_A1_A2_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2456411
NS_A1_A2_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2445322, upper bound: 817.2449159
NS_A1_A2_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2445322, upper bound: 817.2453182
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2484142, upper bound: 817.2452156
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2456384, upper bound: 817.2450553
NS_A2_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2466165, upper bound: 817.2498619
NS_A2_B1_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 3.04
Output dim: 0, lower bound: -817.2456384, upper bound: 817.2467862

## BFS NS instance: NS_A1_A1_A1_A1_B1

### Backsubstitution after applying NS history:
0: -139.8680878, 557.6105957, -158.2226562, 636.4209595, -776.2890015, 715.8332520
1: -173.1533661, 630.2206421, -195.7023926, 719.3157959, -892.4691772, 825.9230347
2: -181.3994293, 639.3145142, -205.3563385, 728.8754272, -910.2748413, 844.6708374
3: -286.9536743, 674.6697388, -325.7484741, 769.5800171, -1056.5335693, 1000.4182129
4: -291.6203003, 649.4708862, -331.2698364, 740.2111816, -1031.8312988, 980.7407227

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_A1_A1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2416071, upper bound: 817.2339295
time: 0.67 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2416071, upper bound: 817.2341786
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2

### Backsubstitution after applying NS history:
0: -139.8680878, 557.6105957, -167.6219482, 673.3251953, -813.1932373, 725.2325439
1: -173.1533661, 630.2206421, -207.3879852, 760.8442993, -933.9976196, 837.6085815
2: -181.3994293, 639.3145142, -217.5605774, 771.2716675, -952.6710815, 856.8751221
3: -286.9536743, 674.6697388, -344.9789124, 814.0966187, -1101.0500488, 1019.6486816
4: -291.6203003, 649.4708862, -350.6948242, 783.3265381, -1074.9467773, 1000.1657104

Time for backsubstitution: 1.49 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A1_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_A1_A1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2416071, upper bound: 817.2339295
time: 0.71 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2416071, upper bound: 817.2341786
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -152.5931091, 611.8168335, -767.1249390, 774.5145264
1: -192.1896210, 702.8735962, -188.8194427, 691.4993286, -883.6889038, 891.6930542
2: -201.3675842, 712.5927124, -198.1996155, 701.1909180, -902.5584717, 910.7923584
3: -319.2729492, 752.2887573, -313.7412720, 739.9826050, -1059.2556152, 1066.0299072
4: -324.6016541, 723.6882324, -318.9634399, 712.4406738, -1037.0422363, 1042.6513672

Time for backsubstitution: 1.50 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A1_A2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2272484
time: 0.79 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2272484
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -168.7872162, 678.4603271, -833.7684937, 790.7085571
1: -192.1896210, 702.8735962, -208.7705078, 766.7617188, -958.9512939, 911.6441040
2: -201.3675842, 712.5927124, -219.0707245, 777.1583252, -978.5258789, 931.6633911
3: -319.2729492, 752.2887573, -347.5421448, 820.5971680, -1139.8701172, 1099.8306885
4: -324.6016541, 723.6882324, -353.4882507, 789.3981934, -1113.9998779, 1077.1762695

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A1_A2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2276261
time: 0.71 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2276261
time: 0.78 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B1

### Backsubstitution after applying NS history:
0: -143.8092804, 574.0617676, -158.5817108, 637.9118652, -781.7210693, 732.6434937
1: -178.0216064, 648.7533569, -196.1628723, 721.0007324, -899.0223389, 844.9161377
2: -186.5074158, 658.0904541, -205.8115997, 730.5714722, -917.0788574, 863.9019165
3: -295.1674500, 694.4026489, -326.5025024, 771.3489380, -1066.5163574, 1020.9051514
4: -299.9495239, 668.4326782, -332.0022278, 741.8901978, -1041.8394775, 1000.4349365

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_A2_A1_B1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2304024
time: 0.73 seconds

## Relational analysis of NS_A1_A1_A2_A1_B1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2313737
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B2

### Backsubstitution after applying NS history:
0: -143.8092804, 574.0617676, -168.0844421, 675.2583618, -819.0676270, 742.1462402
1: -178.0216064, 648.7533569, -207.9768982, 763.0317383, -941.0532837, 856.7301636
2: -186.5074158, 658.0904541, -218.1584778, 773.4871826, -959.9946289, 876.2489014
3: -295.1674500, 694.4026489, -345.9619141, 816.4398193, -1111.6072998, 1040.3643799
4: -299.9495239, 668.4326782, -351.6917725, 785.5378418, -1085.4873047, 1020.1244507

Time for backsubstitution: 1.51 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A1_A2_A1_B2_B1

### Relational analysis result of NS_A1_A1_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2304024
time: 0.73 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2313737
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_A2_A2_B2

### Backsubstitution after applying NS history:
0: -158.3521729, 634.9489136, -169.1327667, 679.9418945, -838.2940063, 804.0815430
1: -195.9344635, 717.5258179, -209.2111816, 768.4354248, -964.3698730, 926.7369995
2: -205.3063965, 727.3727417, -219.5114288, 778.8358765, -984.1422729, 946.8841553
3: -325.6563110, 767.8197021, -348.2723694, 822.3573608, -1148.0136719, 1116.0919189
4: -331.0854797, 738.4993286, -354.2109680, 791.0524902, -1122.1378174, 1092.7100830

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A1_A2_A2_B2_B1

### Relational analysis result of NS_A1_A1_A2_A2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2210498, upper bound: 817.2243383
time: 0.77 seconds

## Relational analysis of NS_A1_A1_A2_A2_B2_B2

### Relational analysis result of NS_A1_A1_A2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2210498, upper bound: 817.2243383
time: 0.75 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B1

### Backsubstitution after applying NS history:
0: -142.6351776, 571.1234741, -161.3270416, 648.9094849, -791.5446777, 732.4503784
1: -176.5977631, 645.6719360, -199.6138763, 733.4785156, -910.0761719, 845.2857666
2: -185.1582794, 654.6025391, -209.4587555, 743.2446289, -928.4028931, 864.0612793
3: -293.1644592, 690.7665405, -332.3055115, 784.7813110, -1077.9458008, 1023.0719604
4: -297.7338257, 665.0226440, -337.8706055, 754.9189453, -1052.6528320, 1002.8932495

Time for backsubstitution: 1.52 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A1_A1_B1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2447706
time: 0.74 seconds

## Relational analysis of NS_A1_A2_A1_A1_B1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2447706
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B2

### Backsubstitution after applying NS history:
0: -142.6351776, 571.1234741, -170.4312439, 684.6336670, -827.2688599, 741.5546875
1: -176.5977631, 645.6719360, -210.9181213, 773.6708374, -950.2685547, 856.5900879
2: -185.1582794, 654.6025391, -221.2723541, 784.2905884, -969.4487915, 875.8748779
3: -293.1644592, 690.7665405, -350.9089661, 827.9018555, -1121.0662842, 1041.6751709
4: -297.7338257, 665.0226440, -356.7006836, 796.6616211, -1094.3955078, 1021.7233276

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A1_A1_B2_B1

### Relational analysis result of NS_A1_A2_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2454974
time: 0.79 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2_B2

### Relational analysis result of NS_A1_A2_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2465135
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B1

### Backsubstitution after applying NS history:
0: -156.2181702, 627.3008423, -156.3930511, 626.8654175, -783.0835571, 783.6937866
1: -193.3670197, 709.1093140, -193.5710602, 708.5474854, -901.9143677, 902.6803589
2: -202.7233887, 718.6606445, -203.1609344, 718.5297241, -921.2531128, 921.8214722
3: -321.6897888, 758.8978882, -321.6680603, 758.3240356, -1080.0136719, 1080.5655518
4: -327.0349426, 729.9923096, -326.9511108, 730.2100830, -1057.2449951, 1056.9433594

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_A1_A2_B1_B1

### Relational analysis result of NS_A1_A2_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2447709
time: 0.77 seconds

## Relational analysis of NS_A1_A2_A1_A2_B1_B2

### Relational analysis result of NS_A1_A2_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2454987
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B2

### Backsubstitution after applying NS history:
0: -156.2181702, 627.3008423, -171.2240906, 688.2742310, -844.4924316, 798.5249023
1: -193.3670197, 709.1093140, -211.8401794, 777.8964233, -971.2633057, 920.9494629
2: -202.7233887, 718.6606445, -222.3001099, 788.4634399, -991.1868286, 940.9607544
3: -321.6897888, 758.8978882, -352.7009583, 832.5778198, -1154.2675781, 1111.5983887
4: -327.0349426, 729.9923096, -358.7004700, 800.9963379, -1128.0311279, 1088.6927490

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_A1_A2_B2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2447709
time: 0.79 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2456520
time: 0.77 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B1

### Backsubstitution after applying NS history:
0: -145.9492340, 584.7414551, -161.6886444, 650.4089355, -796.3581543, 746.4301147
1: -180.7044830, 661.0221558, -200.0760498, 735.1731567, -915.8775024, 861.0980225
2: -189.4848633, 670.1931763, -209.9181671, 744.9511719, -934.4359741, 880.1113281
3: -300.1191101, 707.2645264, -333.0649719, 786.5628052, -1086.6816406, 1040.3294678
4: -304.8388977, 680.8472290, -338.6102905, 756.6113281, -1061.4499512, 1019.4575195

Time for backsubstitution: 1.55 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_A2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A2_A1_B1_B1

### Relational analysis result of NS_A1_A2_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2444322
time: 0.92 seconds

## Relational analysis of NS_A1_A2_A2_A1_B1_B2

### Relational analysis result of NS_A1_A2_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2444322
time: 0.70 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B2

### Backsubstitution after applying NS history:
0: -145.9492340, 584.7414551, -170.9046326, 686.6140747, -832.5632935, 755.6461182
1: -180.7044830, 661.0221558, -211.5225220, 775.9132690, -956.6176758, 872.5446167
2: -189.4848633, 670.1931763, -221.8856354, 786.5611572, -976.0459595, 892.0787964
3: -300.1191101, 707.2645264, -351.9193115, 830.3026733, -1130.4217529, 1059.1835938
4: -304.8388977, 680.8472290, -357.7225037, 798.9312744, -1103.7701416, 1038.5697021

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A1_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_A2_A1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A1_A2_A2_A1_B2_B1

### Relational analysis result of NS_A1_A2_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2451598
time: 0.77 seconds

## Relational analysis of NS_A1_A2_A2_A1_B2_B2

### Relational analysis result of NS_A1_A2_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2456411
time: 0.84 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B1

### Backsubstitution after applying NS history:
0: -158.8125610, 638.4288330, -157.1280060, 629.8989868, -788.7114868, 795.5567017
1: -196.5520020, 721.6295166, -194.4946136, 711.9769897, -908.5289917, 916.1241455
2: -206.0884094, 731.2836914, -204.1071930, 722.0003662, -928.0887451, 935.3908081
3: -327.1409607, 772.1646118, -323.2081299, 761.9907837, -1089.1317139, 1095.3725586
4: -332.5650330, 742.6354980, -328.5093994, 733.7050171, -1066.2700195, 1071.1447754

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_A2_A2_B1_B1

### Relational analysis result of NS_A1_A2_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2441908
time: 0.75 seconds

## Relational analysis of NS_A1_A2_A2_A2_B1_B2

### Relational analysis result of NS_A1_A2_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2449159
time: 0.83 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B2

### Backsubstitution after applying NS history:
0: -158.8125610, 638.4288330, -171.5853119, 689.8228149, -848.6353760, 810.0140381
1: -196.5520020, 721.6295166, -212.3009949, 779.6470337, -976.1990356, 933.9304199
2: -206.0884094, 731.2836914, -222.7634735, 790.2212524, -996.3096924, 954.0471191
3: -327.1409607, 772.1646118, -353.4685364, 834.4232178, -1161.5640869, 1125.6326904
4: -332.5650330, 742.6354980, -359.4571838, 802.7391357, -1135.3041992, 1102.0925293

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 44

## Relational analysis of NS_A1_A2_A2_A2_B2_B1

### Relational analysis result of NS_A1_A2_A2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2441908
time: 0.78 seconds

## Relational analysis of NS_A1_A2_A2_A2_B2_B2

### Relational analysis result of NS_A1_A2_A2_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2449159
time: 0.91 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -165.4939880, 663.3164062, -161.7583923, 650.1112671, -815.6052246, 825.0747070
1: -204.7615967, 749.6140137, -200.0273743, 734.8194580, -939.5809937, 949.6413574
2: -214.7211456, 759.8950195, -209.8936157, 744.4953003, -959.2164307, 969.7885132
3: -340.3456116, 802.1063232, -332.9267578, 786.0782471, -1126.4238281, 1135.0330811
4: -345.8139343, 771.9240112, -338.4233704, 756.1741333, -1101.9877930, 1110.3472900

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2472880, upper bound: 817.2447433
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2479844, upper bound: 817.2447781
time: 0.87 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -168.1442413, 675.0761108, -162.1594849, 651.8029175, -819.9470825, 837.2355957
1: -208.0101318, 762.8388672, -200.5368805, 736.7336426, -944.7437744, 963.3757324
2: -218.1670685, 773.2105713, -210.4062347, 746.4187622, -964.5857544, 983.6167603
3: -345.9537048, 816.0511475, -333.7708435, 788.0932007, -1134.0466309, 1149.8220215
4: -351.5466614, 785.2005615, -339.2527771, 758.0814819, -1109.6281738, 1124.4532471

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2449110, upper bound: 817.2445216
time: 0.65 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2453266, upper bound: 817.2446064
time: 1.17 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -170.3059692, 683.7305298, -165.4939880, 663.3164062, -833.6223755, 849.2244873
1: -210.6697235, 772.6364746, -204.7615967, 749.6140137, -960.2837524, 977.3980713
2: -221.0205078, 783.1354980, -214.7211456, 759.8950195, -980.9154663, 997.8566284
3: -350.4467773, 826.6484375, -340.3456116, 802.1063232, -1152.5528564, 1166.9940186
4: -356.1589966, 795.4536133, -345.8139343, 771.9240112, -1128.0830078, 1141.2673340

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B1_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2464825, upper bound: 817.2495263
time: 0.71 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_B2

### Relational analysis result of NS_A2_B1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2468393, upper bound: 817.2498619
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -170.8119965, 685.8300781, -168.1442413, 675.0761108, -845.8881226, 853.9743042
1: -211.3121185, 775.0186157, -208.0101318, 762.8388672, -974.1509399, 983.0287476
2: -221.6749268, 785.5433960, -218.1670685, 773.2105713, -994.8854980, 1003.7103882
3: -351.5177307, 829.2013550, -345.9537048, 816.0511475, -1167.5686035, 1175.1546631
4: -357.2480774, 797.8613892, -351.5466614, 785.2005615, -1142.4486084, 1149.4080811

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2462120, upper bound: 817.2454123
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2467659, upper bound: 817.2467862
time: 0.70 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.10 seconds
NS_A1_A1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2416071, upper bound: 817.2339295
NS_A1_A1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2416071, upper bound: 817.2341786
NS_A1_A1_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2416071, upper bound: 817.2339295
NS_A1_A1_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2416071, upper bound: 817.2341786
NS_A1_A1_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2272484
NS_A1_A1_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2272484
NS_A1_A1_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2276261
NS_A1_A1_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2276261
NS_A1_A1_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2304024
NS_A1_A1_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2313737
NS_A1_A1_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2304024
NS_A1_A1_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2223065, upper bound: 817.2313737
NS_A1_A1_A2_A2_B2_B1, status: Status.VERIFIED, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2210498, upper bound: 817.2243383
NS_A1_A1_A2_A2_B2_B2, status: Status.VERIFIED, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2210498, upper bound: 817.2243383
NS_A1_A2_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2447706
NS_A1_A2_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2447706
NS_A1_A2_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2454974
NS_A1_A2_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2482656, upper bound: 817.2465135
NS_A1_A2_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2447709
NS_A1_A2_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2454987
NS_A1_A2_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2447709
NS_A1_A2_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2456520
NS_A1_A2_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2444322
NS_A1_A2_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2444322
NS_A1_A2_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2451598
NS_A1_A2_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2456411
NS_A1_A2_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2441908
NS_A1_A2_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2449159
NS_A1_A2_A2_A2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2441908
NS_A1_A2_A2_A2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2449159
NS_A2_B1_A1_B1_A1_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2472880, upper bound: 817.2447433
NS_A2_B1_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2479844, upper bound: 817.2447781
NS_A2_B1_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2449110, upper bound: 817.2445216
NS_A2_B1_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2453266, upper bound: 817.2446064
NS_A2_B1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2464825, upper bound: 817.2495263
NS_A2_B1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2468393, upper bound: 817.2498619
NS_A2_B1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2462120, upper bound: 817.2454123
NS_A2_B1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 6, time: 3.10
Output dim: 0, lower bound: -817.2467659, upper bound: 817.2467862

## BFS NS instance: NS_A1_A1_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -139.8680878, 557.6105957, -144.8664093, 580.7291260, -720.5972290, 702.4769897
1: -173.1533661, 630.2206421, -179.2762604, 656.4140625, -829.5673218, 809.4968872
2: -181.3994293, 639.3145142, -188.0700531, 665.5084229, -846.9078369, 827.3845825
3: -286.9536743, 674.6697388, -297.8073120, 702.3106079, -989.2642212, 972.4770508
4: -291.6203003, 649.4708862, -302.6359558, 676.0338745, -967.6541748, 952.1068115

Time for backsubstitution: 1.53 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2408333, upper bound: 817.2361376
time: 0.69 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2425590, upper bound: 817.2346657
time: 0.75 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -139.8680878, 557.6105957, -160.8954620, 646.9687500, -786.8367920, 718.5060425
1: -173.1533661, 630.2206421, -199.0046539, 731.1917114, -904.3449097, 829.2252197
2: -181.3994293, 639.3145142, -208.7214813, 740.9992676, -922.3986816, 848.0360107
3: -286.9536743, 674.6697388, -331.2548828, 782.3867798, -1069.3402100, 1005.9246216
4: -291.6203003, 649.4708862, -336.8600159, 752.5268555, -1044.1469727, 986.3309326

Time for backsubstitution: 1.54 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2408333, upper bound: 817.2361376
time: 0.67 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2425590, upper bound: 817.2346657
time: 0.73 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -139.8680878, 557.6105957, -155.9306641, 624.2153931, -764.0833740, 713.5412598
1: -173.1533661, 630.2206421, -193.0264587, 705.4248047, -878.5781860, 823.2470703
2: -181.3994293, 639.3145142, -202.4662781, 715.5625610, -896.9619751, 841.7807617
3: -286.9536743, 674.6697388, -320.5224915, 755.1050415, -1042.0584717, 995.1922607
4: -291.6203003, 649.4708862, -325.7439270, 727.1387939, -1018.7590942, 975.2148438

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2399980, upper bound: 817.2323809
time: 0.67 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2415529, upper bound: 817.2337858
time: 0.73 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -139.8680878, 557.6105957, -170.2124329, 683.4532471, -823.3212891, 727.8229980
1: -173.1533661, 630.2206421, -210.5922699, 772.3172607, -945.4705811, 840.8128052
2: -181.3994293, 639.3145142, -220.8510284, 782.9353027, -964.3347168, 860.1655273
3: -286.9536743, 674.6697388, -350.3525696, 826.5365601, -1113.4902344, 1025.0223389
4: -291.6203003, 649.4708862, -356.2004700, 795.1864624, -1086.8065186, 1005.6712036

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2399980, upper bound: 817.2334600
time: 0.76 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2415529, upper bound: 817.2337982
time: 0.67 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -144.8664093, 580.7291260, -736.0372925, 766.7877808
1: -192.1896210, 702.8735962, -179.2762604, 656.4140625, -848.6035767, 882.1498413
2: -201.3675842, 712.5927124, -188.0700531, 665.5084229, -866.8759155, 900.6627808
3: -319.2729492, 752.2887573, -297.8073120, 702.3106079, -1021.5835571, 1050.0960693
4: -324.6016541, 723.6882324, -302.6359558, 676.0338745, -1000.6354980, 1026.3239746

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2323610, upper bound: 817.2170408
time: 0.73 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2359973, upper bound: 817.2269121
time: 0.67 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -155.9306641, 624.2153931, -779.5234985, 777.8520508
1: -192.1896210, 702.8735962, -193.0264587, 705.4248047, -897.6144409, 895.9000244
2: -201.3675842, 712.5927124, -202.4662781, 715.5625610, -916.9300537, 915.0589600
3: -319.2729492, 752.2887573, -320.5224915, 755.1050415, -1074.3778076, 1072.8109131
4: -324.6016541, 723.6882324, -325.7439270, 727.1387939, -1051.7403564, 1049.4318848

Time for backsubstitution: 1.56 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2272484
time: 0.72 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2272484
time: 0.73 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -160.8954620, 646.9687500, -802.2769165, 782.8168335
1: -192.1896210, 702.8735962, -199.0046539, 731.1917114, -923.3811646, 901.8782349
2: -201.3675842, 712.5927124, -208.7214813, 740.9992676, -942.3667603, 921.3142090
3: -319.2729492, 752.2887573, -331.2548828, 782.3867798, -1101.6595459, 1083.5435791
4: -324.6016541, 723.6882324, -336.8600159, 752.5268555, -1077.1284180, 1060.5477295

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2366038, upper bound: 817.2276261
time: 0.82 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2366038, upper bound: 817.2276261
time: 0.76 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -170.2124329, 683.4532471, -838.7614136, 792.1337891
1: -192.1896210, 702.8735962, -210.5922699, 772.3172607, -964.5068359, 913.4658203
2: -201.3675842, 712.5927124, -220.8510284, 782.9353027, -984.3027954, 933.4437256
3: -319.2729492, 752.2887573, -350.3525696, 826.5365601, -1145.8095703, 1102.6411133
4: -324.6016541, 723.6882324, -356.2004700, 795.1864624, -1119.7878418, 1079.8883057

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2366038, upper bound: 817.2276261
time: 0.74 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2366038, upper bound: 817.2276261
time: 0.65 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -143.8092804, 574.0617676, -145.7084198, 584.1345215, -727.9437866, 719.7701416
1: -178.0216064, 648.7533569, -180.3362122, 660.2719116, -838.2934570, 829.0895996
2: -186.5074158, 658.0904541, -189.1623993, 669.4173584, -855.9248047, 847.2527466
3: -295.1674500, 694.4026489, -299.5663757, 706.4439697, -1001.6113892, 993.9689941
4: -299.9495239, 668.4326782, -304.4122009, 679.9979248, -979.9474487, 972.8448486

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2227678, upper bound: 817.2340558
time: 0.67 seconds

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2225982, upper bound: 817.2327929
time: 0.79 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -143.8092804, 574.0617676, -161.3045959, 648.6733398, -792.4825439, 735.3663330
1: -178.0216064, 648.7533569, -199.5246429, 733.1229858, -911.1445923, 848.2778931
2: -186.5074158, 658.0904541, -209.2442169, 742.9367065, -929.4440918, 867.3344727
3: -295.1674500, 694.4026489, -332.1171265, 784.4218750, -1079.5893555, 1026.5195312
4: -299.9495239, 668.4326782, -337.7105408, 754.4503784, -1054.3997803, 1006.1431885

Time for backsubstitution: 1.57 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2227678, upper bound: 817.2340558
time: 0.77 seconds

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B2

### Relational analysis result of NS_A1_A1_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2225982, upper bound: 817.2327929
time: 0.89 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -143.8092804, 574.0617676, -156.9568939, 628.3981934, -772.2074585, 731.0186768
1: -178.0216064, 648.7533569, -194.3139496, 710.1589355, -888.1805420, 843.0671997
2: -186.5074158, 658.0904541, -203.7978668, 720.3601074, -906.8675537, 861.8882446
3: -295.1674500, 694.4026489, -322.6756592, 760.2012329, -1055.3686523, 1017.0782471
4: -299.9495239, 668.4326782, -327.9362183, 731.9968872, -1031.9464111, 996.3688965

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2221980, upper bound: 817.2277140
time: 0.67 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2220352, upper bound: 817.2302108
time: 0.79 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -143.8092804, 574.0617676, -170.7337494, 685.6262207, -829.4354858, 744.7955322
1: -178.0216064, 648.7533569, -211.2546082, 774.7800903, -952.8016357, 860.0078735
2: -186.5074158, 658.0904541, -221.5271759, 785.4255371, -971.9329834, 879.6175537
3: -295.1674500, 694.4026489, -351.4607239, 829.1744995, -1124.3419189, 1045.8634033
4: -299.9495239, 668.4326782, -357.3258667, 797.6792603, -1097.6287842, 1025.7585449

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A2_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2221980, upper bound: 817.2303405
time: 0.71 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2220352, upper bound: 817.2306954
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -142.6351776, 571.1234741, -149.3582001, 598.6764526, -741.3115845, 720.4816895
1: -176.5977631, 645.6719360, -184.8743591, 676.7410278, -853.3386841, 830.5460815
2: -185.1582794, 654.6025391, -193.9218750, 686.1345215, -871.2927856, 848.5243530
3: -293.1644592, 690.7665405, -307.1614685, 724.0966797, -1017.2611084, 997.9279785
4: -297.7338257, 665.0226440, -312.0291138, 697.1099243, -994.8436890, 977.0517578

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_A1_A1_B1_B1_A1

### Relational analysis result of NS_A1_A2_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2454637, upper bound: 817.2372954
time: 0.70 seconds

## Relational analysis of NS_A1_A2_A1_A1_B1_B1_A2

### Relational analysis result of NS_A1_A2_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2454759, upper bound: 817.2372075
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B1_B2

### Backsubstitution after applying NS history:
0: -142.6351776, 571.1234741, -164.0529175, 659.6337280, -802.2689209, 735.1762695
1: -176.5977631, 645.6719360, -202.9769745, 745.5619507, -922.1597290, 848.6488647
2: -185.1582794, 654.6025391, -212.8902588, 755.5803833, -940.7386475, 867.4927979
3: -293.1644592, 690.7665405, -337.9218750, 797.8280640, -1090.9924316, 1028.6883545
4: -297.7338257, 665.0226440, -343.5617676, 767.4690552, -1065.2028809, 1008.5843506

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A1

### Relational analysis result of NS_A1_A2_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2458293, upper bound: 817.2394551
time: 0.88 seconds

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A2

### Relational analysis result of NS_A1_A2_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2460278, upper bound: 817.2394535
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -142.6351776, 571.1234741, -159.9007721, 639.9968872, -782.6320801, 731.0241699
1: -176.5977631, 645.6719360, -197.9869080, 723.3015747, -899.8993530, 843.6588135
2: -185.1582794, 654.6025391, -207.6448517, 733.7279663, -918.8862305, 862.2473755
3: -293.1644592, 690.7665405, -328.7914429, 774.3292236, -1067.4936523, 1019.5578613
4: -297.7338257, 665.0226440, -334.0815735, 745.7349854, -1043.4687500, 999.1041870

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_A1_A1_B2_B1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2491722, upper bound: 817.2421408
time: 0.95 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2_B1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2471666, upper bound: 817.2380297
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B2_B2

### Backsubstitution after applying NS history:
0: -142.6351776, 571.1234741, -172.8744965, 694.2057495, -836.8408813, 743.9979858
1: -176.5977631, 645.6719360, -213.9468079, 784.5135498, -961.1112671, 859.6186523
2: -185.1582794, 654.6025391, -224.3763428, 795.3225708, -980.4808350, 878.9786987
3: -293.1644592, 690.7665405, -355.9911194, 839.6609497, -1132.8254395, 1046.7575684
4: -297.7338257, 665.0226440, -361.8907471, 807.8915405, -1105.6253662, 1026.9133301

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_A1_A1_B2_B2_A1

### Relational analysis result of NS_A1_A2_A1_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2483440, upper bound: 817.2414012
time: 0.70 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2_B2_A2

### Relational analysis result of NS_A1_A2_A1_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2492810, upper bound: 817.2414146
time: 0.62 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B1_B1

### Backsubstitution after applying NS history:
0: -156.2181702, 627.3008423, -149.3582001, 598.6764526, -754.8945923, 776.6590576
1: -193.3670197, 709.1093140, -184.8743591, 676.7410278, -870.1079102, 893.9835815
2: -202.7233887, 718.6606445, -193.9218750, 686.1345215, -888.8579102, 912.5824585
3: -321.6897888, 758.8978882, -307.1614685, 724.0966797, -1045.7864990, 1066.0590820
4: -327.0349426, 729.9923096, -312.0291138, 697.1099243, -1024.1446533, 1042.0214844

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_A1_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_A1_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_A1_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2486930, upper bound: 817.2451756
time: 0.75 seconds

## Relational analysis of NS_A1_A2_A1_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_A1_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2451756
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B1_B2

### Backsubstitution after applying NS history:
0: -156.2181702, 627.3008423, -159.9007721, 639.9968872, -796.2150879, 787.2015381
1: -193.3670197, 709.1093140, -197.9869080, 723.3015747, -916.6685181, 907.0961914
2: -202.7233887, 718.6606445, -207.6448517, 733.7279663, -936.4513550, 926.3054810
3: -321.6897888, 758.8978882, -328.7914429, 774.3292236, -1096.0190430, 1087.6889648
4: -327.0349426, 729.9923096, -334.0815735, 745.7349854, -1072.7698975, 1064.0738525

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_A1_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_A1_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2454987
time: 0.72 seconds

## Relational analysis of NS_A1_A2_A1_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2454987
time: 0.72 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B2_B1

### Backsubstitution after applying NS history:
0: -156.2181702, 627.3008423, -164.0529175, 659.6337280, -815.8519287, 791.3536377
1: -193.3670197, 709.1093140, -202.9769745, 745.5619507, -938.9288330, 912.0863037
2: -202.7233887, 718.6606445, -212.8902588, 755.5803833, -958.3035889, 931.5508423
3: -321.6897888, 758.8978882, -337.9218750, 797.8280640, -1119.5177002, 1096.8194580
4: -327.0349426, 729.9923096, -343.5617676, 767.4690552, -1094.5037842, 1073.5540771

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2447709
time: 1.00 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_A1_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_A1_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_A1_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2447709
time: 0.73 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2_B1_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2447709
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_A1_A2_B2_B2

### Backsubstitution after applying NS history:
0: -156.2181702, 627.3008423, -172.8744965, 694.2057495, -850.4239502, 800.1753540
1: -193.3670197, 709.1093140, -213.9468079, 784.5135498, -977.8804321, 923.0561523
2: -202.7233887, 718.6606445, -224.3763428, 795.3225708, -998.0459595, 943.0368042
3: -321.6897888, 758.8978882, -355.9911194, 839.6609497, -1161.3507080, 1114.8889160
4: -327.0349426, 729.9923096, -361.8907471, 807.8915405, -1134.9262695, 1091.8830566

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_A1_A2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2453935
time: 0.96 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2_B2_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_A1_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_A1_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2456520
time: 0.77 seconds

## Relational analysis of NS_A1_A2_A1_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_A1_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2456520
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -145.9492340, 584.7414551, -150.2556152, 602.3087158, -748.2579346, 734.9970703
1: -180.7044830, 661.0221558, -186.0011139, 680.8543701, -861.5587769, 847.0232544
2: -189.4848633, 670.1931763, -195.0844727, 690.2998047, -879.7846680, 865.2776489
3: -300.1191101, 707.2645264, -309.0311279, 728.5015869, -1028.6207275, 1016.2956543
4: -304.8388977, 680.8472290, -313.9205627, 701.3300171, -1006.1689453, 994.7677612

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_A2_A1_B1_B1_A1

### Relational analysis result of NS_A1_A2_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2394320, upper bound: 817.2366804
time: 0.66 seconds

## Relational analysis of NS_A1_A2_A2_A1_B1_B1_A2

### Relational analysis result of NS_A1_A2_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2361106, upper bound: 817.2362446
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -145.9492340, 584.7414551, -164.4541473, 661.3134155, -807.2626343, 749.1956177
1: -180.7044830, 661.0221558, -203.4870758, 747.4650879, -928.1694336, 864.5091553
2: -189.4848633, 670.1931763, -213.4019470, 757.4885254, -946.9733276, 883.5950317
3: -300.1191101, 707.2645264, -338.7692261, 799.8345947, -1099.9537354, 1046.0334473
4: -304.8388977, 680.8472290, -344.3956299, 769.3630371, -1074.2016602, 1025.2427979

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_A2_A1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_A2_A1_B1_B2_B1

### Relational analysis result of NS_A1_A2_A2_A1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2444322
time: 0.72 seconds

## Relational analysis of NS_A1_A2_A2_A1_B1_B2_B2

### Relational analysis result of NS_A1_A2_A2_A1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2444322
time: 0.66 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -145.9492340, 584.7414551, -160.9717255, 644.3723145, -790.3214722, 745.7131958
1: -180.7044830, 661.0221558, -199.3277893, 728.2512817, -908.9557495, 860.3499146
2: -189.4848633, 670.1931763, -209.0320129, 738.7409668, -928.2258301, 879.2252197
3: -300.1191101, 707.2645264, -331.0333557, 779.6487427, -1079.7677002, 1038.2976074
4: -304.8388977, 680.8472290, -336.3611145, 750.8090820, -1055.6478271, 1017.2083740

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_A2_A1_B2_B1_B1

### Relational analysis result of NS_A1_A2_A2_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2449390, upper bound: 817.2451598
time: 0.73 seconds

## Relational analysis of NS_A1_A2_A2_A1_B2_B1_B2

### Relational analysis result of NS_A1_A2_A2_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2449390, upper bound: 817.2451598
time: 0.71 seconds

## BFS NS instance: NS_A1_A2_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -145.9492340, 584.7414551, -173.3894958, 696.3698730, -842.3190918, 758.1309814
1: -180.7044830, 661.0221558, -214.6022034, 786.9650269, -967.6693726, 875.6243286
2: -189.4848633, 670.1931763, -225.0442352, 797.8011475, -987.2859497, 895.2374268
3: -300.1191101, 707.2645264, -357.0906982, 842.2854004, -1142.4045410, 1064.3551025
4: -304.8388977, 680.8472290, -363.0037231, 810.3731079, -1115.2116699, 1043.8509521

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_A2_A1_B2_B2_A1

### Relational analysis result of NS_A1_A2_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2409991, upper bound: 817.2400711
time: 0.72 seconds

## Relational analysis of NS_A1_A2_A2_A1_B2_B2_A2

### Relational analysis result of NS_A1_A2_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2409435, upper bound: 817.2402258
time: 0.76 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -158.8125610, 638.4288330, -150.2556152, 602.3087158, -761.1212769, 788.6843262
1: -196.5520020, 721.6295166, -186.0011139, 680.8543701, -877.4063721, 907.6306152
2: -206.0884094, 731.2836914, -195.0844727, 690.2998047, -896.3881836, 926.3681641
3: -327.1409607, 772.1646118, -309.0311279, 728.5015869, -1055.6425781, 1081.1953125
4: -332.5650330, 742.6354980, -313.9205627, 701.3300171, -1033.8950195, 1056.5559082

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_A2_A2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_A2_A2_B1_B1_B1

### Relational analysis result of NS_A1_A2_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2444322, upper bound: 817.2445928
time: 0.64 seconds

## Relational analysis of NS_A1_A2_A2_A2_B1_B1_B2

### Relational analysis result of NS_A1_A2_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2444322, upper bound: 817.2445928
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -158.8125610, 638.4288330, -160.9717255, 644.3723145, -803.1847534, 799.4005737
1: -196.5520020, 721.6295166, -199.3277893, 728.2512817, -924.8032837, 920.9572754
2: -206.0884094, 731.2836914, -209.0320129, 738.7409668, -944.8293457, 940.3156738
3: -327.1409607, 772.1646118, -331.0333557, 779.6487427, -1106.7896729, 1103.1976318
4: -332.5650330, 742.6354980, -336.3611145, 750.8090820, -1083.3741455, 1078.9965820

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A1_A2_A2_A2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_A2_A2_B1_B2_B1

### Relational analysis result of NS_A1_A2_A2_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2444322, upper bound: 817.2449159
time: 0.72 seconds

## Relational analysis of NS_A1_A2_A2_A2_B1_B2_B2

### Relational analysis result of NS_A1_A2_A2_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2444322, upper bound: 817.2449159
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B2_B1

### Backsubstitution after applying NS history:
0: -158.8125610, 638.4288330, -164.4541473, 661.3134155, -820.1259155, 802.8829346
1: -196.5520020, 721.6295166, -203.4870758, 747.4650879, -944.0170898, 925.1165771
2: -206.0884094, 731.2836914, -213.4019470, 757.4885254, -963.5769043, 944.6855469
3: -327.1409607, 772.1646118, -338.7692261, 799.8345947, -1126.9755859, 1110.9333496
4: -332.5650330, 742.6354980, -344.3956299, 769.3630371, -1101.9279785, 1087.0310059

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_A2_A2_B2_B1_B1

### Relational analysis result of NS_A1_A2_A2_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2441908
time: 0.81 seconds

## Relational analysis of NS_A1_A2_A2_A2_B2_B1_B2

### Relational analysis result of NS_A1_A2_A2_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2441908
time: 1.22 seconds

## BFS NS instance: NS_A1_A2_A2_A2_B2_B2

### Backsubstitution after applying NS history:
0: -158.8125610, 638.4288330, -173.3894958, 696.3698730, -855.1823120, 811.8182373
1: -196.5520020, 721.6295166, -214.6022034, 786.9650269, -983.5170288, 936.2316895
2: -206.0884094, 731.2836914, -225.0442352, 797.8011475, -1003.8895264, 956.3279419
3: -327.1409607, 772.1646118, -357.0906982, 842.2854004, -1169.4263916, 1129.2550049
4: -332.5650330, 742.6354980, -363.0037231, 810.3731079, -1142.9379883, 1105.6391602

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A2_A2_A2_B2_B2_B1

### Relational analysis result of NS_A1_A2_A2_A2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2449159
time: 0.72 seconds

## Relational analysis of NS_A1_A2_A2_A2_B2_B2_B2

### Relational analysis result of NS_A1_A2_A2_A2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2449159
time: 0.77 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_A1

### Backsubstitution after applying NS history:
0: -145.9358673, 583.4530640, -154.6559906, 622.1539917, -768.0898438, 738.1090088
1: -180.6552582, 659.3679810, -191.2313995, 703.2078247, -883.8630981, 850.5993042
2: -189.3556366, 668.8167725, -200.6938171, 712.3910522, -901.7467041, 869.5104980
3: -299.6396484, 705.5006104, -318.3703003, 752.1727905, -1051.8125000, 1023.8708496
4: -304.2765808, 679.5838013, -323.6348572, 723.5343018, -1027.8109131, 1003.2186279

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2472880, upper bound: 817.2447433
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2472880, upper bound: 817.2447433
time: 0.69 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -160.1390686, 642.3173828, -158.5348511, 637.5753784, -797.7144165, 800.8520508
1: -198.1456757, 725.8140259, -196.0439148, 720.6006470, -918.7463379, 921.8579102
2: -207.6752472, 735.7696533, -205.6547394, 730.0892944, -937.7644043, 941.4242554
3: -329.3888245, 776.5929565, -326.3450623, 770.8134766, -1100.2022705, 1102.9379883
4: -334.6523132, 747.3005981, -331.7024231, 741.4755249, -1076.1278076, 1079.0030518

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2479844, upper bound: 817.2447781
time: 0.69 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2479844, upper bound: 817.2447781
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -149.4819641, 598.4064331, -154.9612732, 623.4419556, -772.9237671, 753.3676147
1: -185.0189056, 676.2152100, -191.6255035, 704.6672363, -889.6861572, 867.8406982
2: -193.9706573, 685.8415527, -201.0800476, 713.8507690, -907.8213501, 886.9216309
3: -307.0677490, 723.5078125, -319.0114441, 753.6949463, -1060.7626953, 1042.5189209
4: -311.9175720, 696.7422485, -324.2526855, 724.9666138, -1036.8841553, 1020.9949341

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2449110, upper bound: 817.2445216
time: 0.67 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2449110, upper bound: 817.2445216
time: 0.70 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -162.6621552, 653.5504150, -158.9310760, 639.2522583, -801.9144287, 812.4814453
1: -201.2398376, 738.4481201, -196.5481110, 722.4982300, -923.7380371, 934.9961548
2: -210.9558563, 748.4818115, -206.1609650, 731.9962158, -942.9519653, 954.6426392
3: -334.7333069, 789.9201660, -327.1815491, 772.8106689, -1107.5435791, 1117.1016846
4: -340.1076965, 759.9780273, -332.5237732, 743.3658447, -1083.4735107, 1092.5017090

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2453266, upper bound: 817.2446064
time: 0.73 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2453266, upper bound: 817.2446064
time: 0.78 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -162.5249786, 653.2747803, -145.9358673, 583.4530640, -745.9779663, 799.2106323
1: -201.0403290, 738.1795044, -180.6552582, 659.3679810, -860.4081421, 918.8347778
2: -210.9074860, 748.1112061, -189.3556366, 668.8167725, -879.7241821, 937.4668579
3: -334.4503174, 789.5715332, -299.6396484, 705.5006104, -1039.9508057, 1089.2111816
4: -339.8681946, 759.7500610, -304.2765808, 679.5838013, -1019.4519653, 1064.0266113

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2454751, upper bound: 817.2493760
time: 0.68 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2454751, upper bound: 817.2495263
time: 0.80 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -166.9227295, 670.4620972, -160.1390686, 642.3173828, -809.2401123, 830.6011963
1: -206.4905701, 757.6047363, -198.1456757, 725.8140259, -932.3045044, 955.7503052
2: -216.5710144, 767.8897095, -207.6752472, 735.7696533, -952.3406982, 975.5648193
3: -343.5204468, 810.5377197, -329.3888245, 776.5929565, -1120.1134033, 1139.9263916
4: -349.0998840, 779.8978882, -334.6523132, 747.3005981, -1096.4005127, 1114.5501709

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2455447, upper bound: 817.2496992
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2455447, upper bound: 817.2498619
time: 0.73 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -163.0183258, 655.3236694, -149.4819641, 598.4064331, -761.4246826, 804.8054810
1: -201.6699066, 740.4958496, -185.0189056, 676.2152100, -877.8851318, 925.5147705
2: -211.5463562, 750.4634399, -193.9706573, 685.8415527, -897.3879395, 944.4338989
3: -335.4977722, 792.0494385, -307.0677490, 723.5078125, -1059.0054932, 1099.1171875
4: -340.9181824, 762.1049194, -311.9175720, 696.7422485, -1037.6604004, 1074.0220947

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 27

## Relational analysis of NS_A2_B1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 17

## Relational analysis of NS_A2_B1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2452599, upper bound: 817.2452599
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2452599, upper bound: 817.2454123
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -167.4407501, 672.6058350, -162.6621552, 653.5504150, -820.9911499, 835.2680054
1: -207.1486664, 760.0365601, -201.2398376, 738.4481201, -945.5968018, 961.2763672
2: -217.2432404, 770.3512573, -210.9558563, 748.4818115, -965.7250366, 981.3071289
3: -344.6196594, 813.1489868, -334.7333069, 789.9201660, -1134.5397949, 1147.8820801
4: -350.2173157, 782.3644409, -340.1076965, 759.9780273, -1110.1950684, 1122.4721680

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: B, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 26

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2452599, upper bound: 817.2455031
time: 0.79 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2452599, upper bound: 817.2467862
time: 0.74 seconds

## Summary of splitting at layer (split count: 6)
- Time for NS candidates: 3.38 seconds
NS_A1_A1_A1_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2408333, upper bound: 817.2361376
NS_A1_A1_A1_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2425590, upper bound: 817.2346657
NS_A1_A1_A1_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2408333, upper bound: 817.2361376
NS_A1_A1_A1_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2425590, upper bound: 817.2346657
NS_A1_A1_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2399980, upper bound: 817.2323809
NS_A1_A1_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2415529, upper bound: 817.2337858
NS_A1_A1_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2399980, upper bound: 817.2334600
NS_A1_A1_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2415529, upper bound: 817.2337982
NS_A1_A1_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2323610, upper bound: 817.2170408
NS_A1_A1_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2359973, upper bound: 817.2269121
NS_A1_A1_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2272484
NS_A1_A1_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2272484
NS_A1_A1_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2366038, upper bound: 817.2276261
NS_A1_A1_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2366038, upper bound: 817.2276261
NS_A1_A1_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2366038, upper bound: 817.2276261
NS_A1_A1_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2366038, upper bound: 817.2276261
NS_A1_A1_A2_A1_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2227678, upper bound: 817.2340558
NS_A1_A1_A2_A1_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2225982, upper bound: 817.2327929
NS_A1_A1_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2227678, upper bound: 817.2340558
NS_A1_A1_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2225982, upper bound: 817.2327929
NS_A1_A1_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2221980, upper bound: 817.2277140
NS_A1_A1_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2220352, upper bound: 817.2302108
NS_A1_A1_A2_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2221980, upper bound: 817.2303405
NS_A1_A1_A2_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2220352, upper bound: 817.2306954
NS_A1_A2_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2454637, upper bound: 817.2372954
NS_A1_A2_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2454759, upper bound: 817.2372075
NS_A1_A2_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2458293, upper bound: 817.2394551
NS_A1_A2_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2460278, upper bound: 817.2394535
NS_A1_A2_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2491722, upper bound: 817.2421408
NS_A1_A2_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2471666, upper bound: 817.2380297
NS_A1_A2_A1_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2483440, upper bound: 817.2414012
NS_A1_A2_A1_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2492810, upper bound: 817.2414146
NS_A1_A2_A1_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2486930, upper bound: 817.2451756
NS_A1_A2_A1_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2451756
NS_A1_A2_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2454987
NS_A1_A2_A1_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2365502, upper bound: 817.2454987
NS_A1_A2_A1_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2447709
NS_A1_A2_A1_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2447709
NS_A1_A2_A1_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2456520
NS_A1_A2_A1_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2481490, upper bound: 817.2456520
NS_A1_A2_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2394320, upper bound: 817.2366804
NS_A1_A2_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2361106, upper bound: 817.2362446
NS_A1_A2_A2_A1_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2444322
NS_A1_A2_A2_A1_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2445928, upper bound: 817.2444322
NS_A1_A2_A2_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2449390, upper bound: 817.2451598
NS_A1_A2_A2_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2449390, upper bound: 817.2451598
NS_A1_A2_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2409991, upper bound: 817.2400711
NS_A1_A2_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2409435, upper bound: 817.2402258
NS_A1_A2_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2444322, upper bound: 817.2445928
NS_A1_A2_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2444322, upper bound: 817.2445928
NS_A1_A2_A2_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2444322, upper bound: 817.2449159
NS_A1_A2_A2_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2444322, upper bound: 817.2449159
NS_A1_A2_A2_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2441908
NS_A1_A2_A2_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2441908
NS_A1_A2_A2_A2_B2_B2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2449159
NS_A1_A2_A2_A2_B2_B2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2441908, upper bound: 817.2449159
NS_A2_B1_A1_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2472880, upper bound: 817.2447433
NS_A2_B1_A1_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2472880, upper bound: 817.2447433
NS_A2_B1_A1_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2479844, upper bound: 817.2447781
NS_A2_B1_A1_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2479844, upper bound: 817.2447781
NS_A2_B1_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2449110, upper bound: 817.2445216
NS_A2_B1_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2449110, upper bound: 817.2445216
NS_A2_B1_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2453266, upper bound: 817.2446064
NS_A2_B1_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2453266, upper bound: 817.2446064
NS_A2_B1_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2454751, upper bound: 817.2493760
NS_A2_B1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2454751, upper bound: 817.2495263
NS_A2_B1_A1_B2_B1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2455447, upper bound: 817.2496992
NS_A2_B1_A1_B2_B1_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2455447, upper bound: 817.2498619
NS_A2_B1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2452599, upper bound: 817.2452599
NS_A2_B1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2452599, upper bound: 817.2454123
NS_A2_B1_A1_B2_B2_B2_A1, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2452599, upper bound: 817.2455031
NS_A2_B1_A1_B2_B2_B2_A2, status: Status.UNKNOWN, split count: 7, time: 3.38
Output dim: 0, lower bound: -817.2452599, upper bound: 817.2467862

## BFS NS instance: NS_A1_A1_A1_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -128.2356262, 510.2983704, -119.9813538, 481.4330444, -609.6687012, 630.2797241
1: -158.8685150, 576.7097778, -148.5902100, 544.2630615, -703.1315918, 725.2999268
2: -166.2545471, 585.1725464, -155.7693634, 551.7017822, -717.9562988, 740.9418945
3: -262.9597168, 617.5349121, -246.3224030, 582.0591431, -845.0187988, 863.8572998
4: -267.2049561, 594.5346680, -250.3466644, 560.6114502, -827.8164062, 844.8813477

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2337687, upper bound: 817.2188357
time: 0.61 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2432947, upper bound: 817.2348604
time: 0.65 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -135.1986694, 539.1900024, -138.5401764, 555.8223267, -691.0208740, 677.7301025
1: -167.3828583, 609.3923950, -171.4481354, 628.2422485, -795.6250610, 780.8404541
2: -175.3115540, 618.1362305, -179.8055573, 636.8909912, -812.2025146, 797.9417725
3: -277.3598633, 652.3344727, -284.8399353, 672.0939941, -949.4538574, 937.1744385
4: -281.8852844, 627.9202271, -289.4407959, 646.9212646, -928.8065186, 917.3610229

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2437389, upper bound: 817.2353700
time: 0.80 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B1_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2437389, upper bound: 817.2353700
time: 0.77 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -128.2356262, 510.2983704, -138.1661072, 555.8941040, -684.1297607, 648.4644165
1: -158.8685150, 576.7097778, -171.0051727, 628.4386597, -787.3071899, 747.7149658
2: -166.2545471, 585.1725464, -179.2425079, 636.7052002, -802.9597168, 764.4149170
3: -262.9597168, 617.5349121, -284.2621765, 672.3363647, -935.2960815, 901.7969971
4: -267.2049561, 594.5346680, -289.1191406, 646.9340210, -914.1389771, 883.6538086

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2374153, upper bound: 817.2333761
time: 0.65 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2399741, upper bound: 817.2335482
time: 0.65 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -135.1986694, 539.1900024, -153.6721497, 618.8181763, -754.0166016, 692.8621826
1: -167.3828583, 609.3923950, -190.0863953, 699.3439941, -866.7267456, 799.4787598
2: -175.3115540, 618.1362305, -199.2630768, 708.6112671, -883.9228516, 817.3992920
3: -277.3598633, 652.3344727, -316.4290771, 748.0761719, -1025.4360352, 968.7635498
4: -281.8852844, 627.9202271, -321.7326965, 719.4620972, -1001.3473511, 949.6528931

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2385226, upper bound: 817.2216016
time: 0.95 seconds

## Relational analysis of NS_A1_A1_A1_A1_B1_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2385226, upper bound: 817.2346657
time: 0.67 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -128.2356262, 510.2983704, -131.3693695, 526.6701660, -654.9057617, 641.6677246
1: -158.8685150, 576.7097778, -162.6783142, 595.3767700, -754.2453003, 739.3880615
2: -166.2545471, 585.1725464, -170.6777649, 603.6530762, -769.9075928, 755.8502197
3: -262.9597168, 617.5349121, -269.7239990, 636.9940796, -899.9537964, 887.2589111
4: -267.2049561, 594.5346680, -274.2881470, 613.5615845, -880.7665405, 868.8228149

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2394767, upper bound: 817.2306968
time: 0.77 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_A1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2400974, upper bound: 817.2323809
time: 0.70 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B1_A2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2371095, upper bound: 817.2304833
time: 0.76 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -135.1986694, 539.1900024, -148.4726257, 595.1577759, -730.3562622, 687.6625977
1: -167.3828583, 609.3923950, -183.8025055, 672.5553589, -839.9381104, 793.1948242
2: -175.3115540, 618.1362305, -192.7098236, 682.0869751, -857.3985596, 810.8460693
3: -277.3598633, 652.3344727, -305.1982117, 719.7038574, -997.0637207, 957.5326538
4: -281.8852844, 627.9202271, -310.1364136, 692.9653931, -974.8505859, 938.0566406

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2415529, upper bound: 817.2337858
time: 0.88 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2415529, upper bound: 817.2337858
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -128.2356262, 510.2983704, -148.0782013, 595.1658325, -723.4014893, 658.3765869
1: -158.8685150, 576.7097778, -183.2761383, 672.7526855, -831.6212158, 759.9859009
2: -166.2545471, 585.1725464, -192.1614227, 681.7573242, -848.0118408, 777.3339233
3: -262.9597168, 617.5349121, -304.5349426, 719.7631226, -982.7228394, 922.0698242
4: -267.2049561, 594.5346680, -309.7248535, 692.6959839, -959.9009399, 904.2595215

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2371587, upper bound: 817.2298382
time: 0.66 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2390602, upper bound: 817.2302928
time: 0.72 seconds

## BFS NS instance: NS_A1_A1_A1_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -135.1986694, 539.1900024, -162.2128448, 652.1326904, -787.3312378, 701.4027710
1: -167.3828583, 609.3923950, -200.7081299, 736.8768921, -904.2597046, 810.1005249
2: -175.3115540, 618.1362305, -210.3705597, 746.8884277, -922.1999512, 828.5067139
3: -277.3598633, 652.3344727, -333.8820496, 788.3830566, -1065.7429199, 986.2164917
4: -281.8852844, 627.9202271, -339.4390259, 758.3731079, -1040.2583008, 967.3591309

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2353373, upper bound: 817.2109179
time: 0.93 seconds

## Relational analysis of NS_A1_A1_A1_A1_B2_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2353373, upper bound: 817.2337982
time: 0.80 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1_B1_B1

### Backsubstitution after applying NS history:
0: -141.7021790, 567.1248169, -119.9813538, 481.4330444, -623.1352539, 687.1060181
1: -175.4900665, 640.9550781, -148.5902100, 544.2630615, -719.7531128, 789.5452271
2: -183.6552582, 649.8423462, -155.7693634, 551.7017822, -735.3570557, 805.6116943
3: -291.1827393, 685.9017334, -246.3224030, 582.0591431, -873.2416992, 932.2241211
4: -295.9773254, 659.8313599, -250.3466644, 560.6114502, -856.5886841, 910.1780396

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2299062, upper bound: 817.2182128
time: 0.66 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2327154, upper bound: 817.2280179
time: 0.68 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1_B1_B2

### Backsubstitution after applying NS history:
0: -150.3732910, 602.5829468, -138.5401764, 555.8223267, -706.1956177, 741.1230469
1: -186.1030579, 680.9848633, -171.4481354, 628.2422485, -814.3453369, 852.4329224
2: -194.9120178, 690.3392944, -179.8055573, 636.8909912, -831.8029785, 870.1448364
3: -309.1254578, 728.7666626, -284.8399353, 672.0939941, -981.2194824, 1013.6065674
4: -314.2771606, 700.9874878, -289.4407959, 646.9212646, -961.1983032, 990.4282227

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B2_A1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1_B2_A1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2186639, upper bound: 817.2189960
time: 0.72 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B1_B2_A2

### Relational analysis result of NS_A1_A1_A1_A2_B1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2360429, upper bound: 817.2275375
time: 0.77 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1_B2_B1

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -151.2386322, 604.4950562, -759.8032227, 773.1600342
1: -192.1896210, 702.8735962, -187.2536011, 683.1697998, -875.3593750, 890.1270752
2: -201.3675842, 712.5927124, -196.3075104, 693.0875854, -894.4551392, 908.9002075
3: -319.2729492, 752.2887573, -310.6746826, 731.3078613, -1050.5808105, 1062.9632568
4: -324.6016541, 723.6882324, -315.6297913, 704.3356934, -1028.9371338, 1039.3177490

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2322296, upper bound: 817.2100392
time: 0.70 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2322296, upper bound: 817.2272484
time: 0.74 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B1_B2_B2

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -154.7453003, 619.5169067, -774.8250732, 776.6666870
1: -192.1896210, 702.8735962, -191.5715179, 700.0864258, -892.2760010, 894.4450684
2: -201.3675842, 712.5927124, -200.8847809, 710.1591187, -911.5266113, 913.4774780
3: -319.2729492, 752.2887573, -318.0680847, 749.3406982, -1068.6136475, 1070.3568115
4: -324.6016541, 723.6882324, -323.2282410, 721.5199585, -1046.1215820, 1046.9162598

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2322296, upper bound: 817.2100392
time: 1.09 seconds

## Relational analysis of NS_A1_A1_A1_A2_B1_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2322296, upper bound: 817.2272484
time: 0.72 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B1_B1

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -155.7063446, 625.2356567, -780.5438232, 777.6277466
1: -192.1896210, 702.8735962, -192.6247559, 706.6767578, -898.8663330, 895.4982910
2: -201.3675842, 712.5927124, -201.9410858, 716.2140503, -917.5815430, 914.5338135
3: -319.2729492, 752.2887573, -320.3985596, 756.1802368, -1075.4531250, 1072.6871338
4: -324.6016541, 723.6882324, -325.7605896, 727.3939209, -1051.9954834, 1049.4484863

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2352865, upper bound: 817.2321822
time: 0.73 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2368444, upper bound: 817.2315020
time: 0.67 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B1_B2

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -158.4537659, 637.1347046, -792.4428711, 780.3751221
1: -192.1896210, 702.8735962, -195.9959717, 720.0535889, -912.2431030, 898.8695068
2: -201.3675842, 712.5927124, -205.4966125, 729.6986084, -931.0661011, 918.0892944
3: -319.2729492, 752.2887573, -326.1708069, 770.3062744, -1089.5791016, 1078.4592285
4: -324.6016541, 723.6882324, -331.5946045, 740.8793945, -1065.4807129, 1055.2828369

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2352865, upper bound: 817.2321822
time: 0.72 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B1_B2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2368444, upper bound: 817.2315020
time: 0.72 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B2_B1

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -165.4893341, 663.4093018, -818.7174683, 787.4107056
1: -192.1896210, 702.8735962, -204.7994232, 749.7128296, -941.9024048, 907.6730347
2: -201.3675842, 712.5927124, -214.6729279, 760.1246338, -961.4921265, 927.2656250
3: -319.2729492, 752.2887573, -340.4559021, 802.4416504, -1121.7144775, 1092.7443848
4: -324.6016541, 723.6882324, -346.0637207, 772.0880737, -1096.6895752, 1069.7518311

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2172564, upper bound: 817.2220768
time: 0.99 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B1_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2360340, upper bound: 817.2271045
time: 0.66 seconds

## BFS NS instance: NS_A1_A1_A1_A2_B2_B2_B2

### Backsubstitution after applying NS history:
0: -155.3081665, 621.9213867, -168.0583038, 674.8718872, -830.1800537, 789.9796143
1: -192.1896210, 702.8735962, -207.9456787, 762.5907593, -954.7803345, 910.8192749
2: -201.3675842, 712.5927124, -218.0143738, 773.0884399, -974.4560547, 930.6070557
3: -319.2729492, 752.2887573, -345.8961487, 816.0226440, -1135.2956543, 1098.1846924
4: -324.6016541, 723.6882324, -351.6223145, 785.0166016, -1109.6182861, 1075.3103027

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B2_B1

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B2_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2172565, upper bound: 817.2221494
time: 0.74 seconds

## Relational analysis of NS_A1_A1_A1_A2_B2_B2_B2_B2

### Relational analysis result of NS_A1_A1_A1_A2_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2360340, upper bound: 817.2271045
time: 0.70 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B1_B1_B1

### Backsubstitution after applying NS history:
0: -132.3080444, 527.1289673, -120.4749756, 483.3989258, -615.7068481, 647.6039429
1: -163.9283905, 595.7002563, -149.2009125, 546.4901123, -710.4185181, 744.9011841
2: -171.5370178, 604.4366455, -156.3958435, 553.9514771, -725.4884644, 760.8323364
3: -271.4551086, 637.8051758, -247.3317566, 584.4462891, -855.9013672, 885.1369629
4: -275.7754822, 614.0665283, -251.3751068, 562.8800659, -838.6554565, 865.4414673

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2228663, upper bound: 817.2341904
time: 0.72 seconds

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B1_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2228663, upper bound: 817.2341904
time: 0.77 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B1_B1_B2

### Backsubstitution after applying NS history:
0: -139.1691589, 555.6784668, -139.3702850, 559.1682129, -698.3374023, 695.0487671
1: -172.2934570, 627.9567871, -172.4945679, 632.0335083, -804.3269653, 800.4512329
2: -180.4516449, 636.9663086, -180.8825989, 640.7295532, -821.1812134, 817.8488770
3: -285.6464233, 672.1310425, -286.5762939, 676.1633911, -961.8096924, 958.7073364
4: -290.2633972, 646.9606323, -291.1940918, 650.8181152, -941.0814819, 938.1547241

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B2_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2227762, upper bound: 817.2336190
time: 0.76 seconds

## Relational analysis of NS_A1_A1_A2_A1_B1_B1_B2_B2

### Relational analysis result of NS_A1_A1_A2_A1_B1_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2227762, upper bound: 817.2336190
time: 0.73 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B1_B2_B1

### Backsubstitution after applying NS history:
0: -132.3080444, 527.1289673, -138.3836060, 556.8335571, -689.1414795, 665.5125732
1: -163.9283905, 595.7002563, -171.2778778, 629.5037231, -793.4320679, 766.9780884
2: -171.5370178, 604.4366455, -179.5129242, 637.7544556, -809.2915039, 783.9495850
3: -271.4551086, 637.8051758, -284.7092285, 673.4426880, -944.8977661, 922.5142822
4: -275.7754822, 614.0665283, -289.5629883, 647.9451294, -923.7205200, 903.6295166

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2227678, upper bound: 817.2340558
time: 0.76 seconds

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2227678, upper bound: 817.2340558
time: 0.74 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B1_B2_B2

### Backsubstitution after applying NS history:
0: -139.1691589, 555.6784668, -154.0308838, 620.3023682, -759.4714966, 709.7092896
1: -172.2934570, 627.9567871, -190.5476532, 701.0272217, -873.3205566, 818.5043945
2: -180.4516449, 636.9663086, -199.7176361, 710.2974854, -890.7491455, 836.6838989
3: -285.6464233, 672.1310425, -317.1846313, 749.8461304, -1035.4925537, 989.3156738
4: -290.2633972, 646.9606323, -322.4707336, 721.1285400, -1011.3919678, 969.4313354

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B2_B1

### Relational analysis result of NS_A1_A1_A2_A1_B1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2225982, upper bound: 817.2327929
time: 0.77 seconds

## Relational analysis of NS_A1_A1_A2_A1_B1_B2_B2_B2

### Relational analysis result of NS_A1_A1_A2_A1_B1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2225982, upper bound: 817.2327929
time: 0.75 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -132.3080444, 527.1289673, -132.1958771, 530.0258789, -662.3338623, 659.3248291
1: -163.9283905, 595.7002563, -163.7114716, 599.1825562, -763.1108398, 759.4115601
2: -171.5370178, 604.4366455, -171.7435455, 607.4943237, -779.0313110, 776.1801758
3: -271.4551086, 637.8051758, -271.4515686, 641.1139526, -912.5690918, 909.2567139
4: -275.7754822, 614.0665283, -276.0827332, 617.4409180, -893.2163086, 890.1492920

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 43

### Candidate
type: B, layer: 1, pos: 0

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 13

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 19

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 0

### Candidate
type: A, layer: 1, pos: 24

### Candidate
type: A, layer: 1, pos: 19

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 2

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 5

### Candidate
type: A, layer: 1, pos: 16

### Candidate
type: B, layer: 1, pos: 37

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 21

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 32

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 42

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 42

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 32

### Candidate
type: B, layer: 1, pos: 40

### Candidate
type: A, layer: 1, pos: 18

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 35

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

### Candidate
type: B, layer: 1, pos: 29

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 35

### Candidate
type: A, layer: 1, pos: 29

### Candidate
type: A, layer: 1, pos: 37

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2220311, upper bound: 817.2240373
time: 0.67 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2222160, upper bound: 817.2277139
time: 0.82 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -139.1691589, 555.6784668, -149.5126801, 599.3835449, -738.5527344, 705.1911011
1: -172.2934570, 627.9567871, -185.1096344, 677.3397827, -849.6331787, 813.0664062
2: -180.4516449, 636.9663086, -194.0594177, 686.9347534, -867.3864136, 831.0257568
3: -285.6464233, 672.1310425, -307.3808289, 724.8636475, -1010.5100098, 979.5118408
4: -290.2633972, 646.9606323, -312.3657532, 697.8765259, -988.1398315, 959.3263550

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B2_B1

### Relational analysis result of NS_A1_A1_A2_A1_B2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2167681, upper bound: 817.2302108
time: 0.70 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2_B1_B2_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2167681, upper bound: 817.2302108
time: 0.77 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B2_B2_B1

### Backsubstitution after applying NS history:
0: -132.3080444, 527.1289673, -148.3711700, 596.3956299, -728.7035522, 675.5001221
1: -163.9283905, 595.7002563, -183.6477661, 674.1492920, -838.0775146, 779.3479004
2: -171.5370178, 604.4366455, -192.5397949, 683.1628418, -854.6998291, 796.9763794
3: -271.4551086, 637.8051758, -305.1548462, 721.2604980, -992.7155762, 942.9600220
4: -275.7754822, 614.0665283, -310.3560791, 694.0855713, -969.8609619, 924.4224854

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B1_B1

### Relational analysis result of NS_A1_A1_A2_A1_B2_B2_B1_B1
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2178109, upper bound: 817.2263856
time: 0.75 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B1_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2_B2_B1_B2
Status: Status.VERIFIED
Output dim: 0, lower bound: -817.2184047, upper bound: 817.2262293
time: 0.80 seconds

## BFS NS instance: NS_A1_A1_A2_A1_B2_B2_B2

### Backsubstitution after applying NS history:
0: -139.1691589, 555.6784668, -162.7689209, 654.4185791, -793.5877686, 718.4473267
1: -172.2934570, 627.9567871, -201.4155884, 739.4691772, -911.7625732, 829.3723145
2: -180.4516449, 636.9663086, -211.0900574, 749.5110474, -929.9627075, 848.0563965
3: -285.6464233, 672.1310425, -335.0611572, 791.1672363, -1076.8135986, 1007.1921997
4: -290.2633972, 646.9606323, -340.6334229, 761.0032959, -1051.2667236, 987.5939331

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 8

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B2_B1

### Relational analysis result of NS_A1_A1_A2_A1_B2_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2192596, upper bound: 817.2306954
time: 0.79 seconds

## Relational analysis of NS_A1_A1_A2_A1_B2_B2_B2_B2

### Relational analysis result of NS_A1_A1_A2_A1_B2_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2192596, upper bound: 817.2306954
time: 0.83 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -116.1015091, 464.8819580, -137.5957642, 551.3506470, -667.4521484, 602.4777222
1: -143.8508301, 525.6766968, -170.4346161, 623.2344971, -767.0853271, 696.1113281
2: -150.7595367, 532.9050293, -178.5432281, 631.9069824, -782.6664429, 711.4482422
3: -238.2851410, 562.3008423, -282.8293762, 666.6336670, -904.9188232, 845.1302490
4: -242.1373749, 541.6639404, -287.0867920, 641.9284668, -884.0658569, 828.7506714

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_A1_A1_B1_B1_A1_A1

### Relational analysis result of NS_A1_A2_A1_A1_B1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2451769, upper bound: 817.2370206
time: 0.73 seconds

## Relational analysis of NS_A1_A2_A1_A1_B1_B1_A1_A2

### Relational analysis result of NS_A1_A2_A1_A1_B1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2464703, upper bound: 817.2370212
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B1_B1_A2

### Backsubstitution after applying NS history:
0: -136.3913879, 546.4343872, -145.0228577, 581.6495361, -718.0408936, 691.4571533
1: -168.8641357, 617.7490234, -179.5103149, 657.4833984, -826.3475342, 797.2593384
2: -177.0017700, 626.2219849, -188.2522888, 666.5522461, -843.5539551, 814.4742432
3: -280.3401184, 660.8504639, -298.2589111, 703.4027100, -983.7427979, 959.1093750
4: -284.7116394, 636.1549072, -302.9614868, 677.1502686, -961.8619385, 939.1163940

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_A1_A1_B1_B1_A2_B1

### Relational analysis result of NS_A1_A2_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2471963, upper bound: 817.2378035
time: 0.92 seconds

## Relational analysis of NS_A1_A2_A1_A1_B1_B1_A2_B2

### Relational analysis result of NS_A1_A2_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2471963, upper bound: 817.2378035
time: 0.66 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B1_B2_A1

### Backsubstitution after applying NS history:
0: -133.9604340, 532.9328003, -154.5698395, 619.8801880, -753.8405762, 687.5026245
1: -165.7510834, 602.4528198, -191.2938080, 700.5690308, -866.3201294, 793.7465210
2: -173.5753479, 611.0177612, -200.4378967, 710.0956421, -883.6710205, 811.4556274
3: -274.7054749, 645.1277466, -318.2267456, 749.9527588, -1024.6582031, 963.3544922
4: -279.0012817, 620.9373169, -323.4373474, 721.2413330, -1000.2426147, 944.3746338

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 33

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2458293, upper bound: 817.2394535
time: 0.69 seconds

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2458293, upper bound: 817.2394535
time: 0.63 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B1_B2_A2

### Backsubstitution after applying NS history:
0: -144.9550629, 579.0523682, -161.0843506, 647.7600708, -792.7150879, 740.1367188
1: -179.5210114, 654.5037842, -199.2863464, 732.1019287, -911.6229248, 853.7901001
2: -187.9488831, 663.8588867, -208.9794464, 741.9539185, -929.9028320, 872.8383179
3: -297.8714294, 700.4893188, -331.7578125, 783.3865356, -1081.2579346, 1032.2470703
4: -302.2611389, 674.5850830, -337.3185425, 753.5676270, -1055.8287354, 1011.9036255

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 13

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 4

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A2_B1

### Relational analysis result of NS_A1_A2_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2457864, upper bound: 817.2386720
time: 0.72 seconds

## Relational analysis of NS_A1_A2_A1_A1_B1_B2_A2_B2

### Relational analysis result of NS_A1_A2_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2449437, upper bound: 817.2350931
time: 0.69 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B2_B1_B1

### Backsubstitution after applying NS history:
0: -131.4631042, 525.5174561, -134.8903961, 540.7289429, -672.1920166, 660.4077759
1: -162.8374329, 594.1063232, -167.0839233, 611.3070679, -774.1445312, 761.1902466
2: -170.4862823, 602.3959351, -175.3117676, 619.8260498, -790.3123169, 777.7077026
3: -269.9375916, 635.5147705, -277.0866089, 654.1219482, -924.0595703, 912.6012573
4: -273.9372253, 611.9931030, -281.7466125, 630.1054688, -904.0427246, 893.7397461

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 17
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 47

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 43

## Relational analysis of NS_A1_A2_A1_A1_B2_B1_B1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2454935, upper bound: 817.2409084
time: 0.72 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2_B1_B1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2392008, upper bound: 817.2409114
time: 0.74 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B2_B1_B2

### Backsubstitution after applying NS history:
0: -138.4326630, 554.5217285, -152.3983459, 610.7094727, -749.1420898, 706.9200439
1: -171.3975372, 626.8944092, -188.7154541, 690.1735229, -861.5710449, 815.6097412
2: -179.6669769, 635.5155029, -197.8352051, 700.0045776, -879.6715698, 833.3507080
3: -284.5261230, 670.6290283, -313.3875732, 738.6646729, -1023.1907959, 984.0166016
4: -288.9519043, 645.5974121, -318.3914490, 711.3195801, -1000.2714844, 963.9888916

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_A1_A1_B2_B1_B2_A1

### Relational analysis result of NS_A1_A2_A1_A1_B2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2471403, upper bound: 817.2380297
time: 0.80 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2_B1_B2_A2

### Relational analysis result of NS_A1_A2_A1_A1_B2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2471403, upper bound: 817.2380297
time: 0.79 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -133.9604340, 532.9328003, -162.8887787, 652.3321533, -786.2925415, 695.8215942
1: -165.7510834, 602.4528198, -201.6273499, 737.0908813, -902.8419800, 804.0802002
2: -173.5753479, 611.0177612, -211.2492981, 747.4358521, -921.0112305, 822.2670288
3: -274.7054749, 645.1277466, -335.1971130, 789.2081909, -1063.9136963, 980.3248291
4: -279.0012817, 620.9373169, -340.7105408, 759.1950073, -1038.1962891, 961.6478271

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 14
type: A, layer: 1, pos: 14
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 25
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 25
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 14

### Candidate
type: A, layer: 1, pos: 14

### Candidate
type: B, layer: 1, pos: 33

## Relational analysis of NS_A1_A2_A1_A1_B2_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 17

### Candidate
type: B, layer: 1, pos: 27

### Candidate
type: B, layer: 1, pos: 4

## Relational analysis of NS_A1_A2_A1_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_A2_A1_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2483170, upper bound: 817.2413770
time: 0.68 seconds

## Relational analysis of NS_A1_A2_A1_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_A2_A1_A1_B2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -817.2483440, upper bound: 817.2414012
time: 0.76 seconds

## BFS NS instance: NS_A1_A2_A1_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -144.9550629, 579.0523682, -170.1239777, 683.2313232, -828.1864014, 749.1762085
1: -179.5210114, 654.5037842, -210.5306854, 772.0726318, -951.5936279, 865.0344849
2: -187.9488831, 663.8588867, -220.7609863, 782.7052612, -970.6541138, 884.6198120
3: -297.8714294, 700.4893188, -350.2791748, 826.2812500, -1124.1527100, 1050.7683105
4: -302.2611389, 674.5850830, -356.0724792, 795.0148926, -1097.2760010, 1030.6575928

Time for backsubstitution: 1.78 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 3.45 + 417.49 = 420.95 seconds
