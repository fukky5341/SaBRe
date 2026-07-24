## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.01171875
execution index: (3, 2, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.37300728


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=15, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=20, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.1989612, 0.1405566, -0.1989612, 0.1405566, -0.3395178, 0.3395178)
1: (-0.1852108, 0.1494017, -0.1852108, 0.1494017, -0.3346125, 0.3346125)
2: (-0.1239861, 0.2424676, -0.1239861, 0.2424676, -0.3664537, 0.3664537)
3: (-0.1087685, 0.3044794, -0.1087685, 0.3044794, -0.4081281, 0.4081281)
4: (-0.1610783, 0.1899182, -0.1610783, 0.1899182, -0.3509965, 0.3509965)
5: (-0.1507930, 0.2258438, -0.1507930, 0.2258438, -0.3766368, 0.3766368)
6: (-0.1878836, 0.1786860, -0.1878836, 0.1786860, -0.3665696, 0.3665696)
7: (0.5364122, 1.0834043, 0.5364122, 1.0834043, -0.5469921, 0.5469921)
8: (-0.1376918, 0.2622380, -0.1376918, 0.2622380, -0.3999298, 0.3999298)
9: (-0.1424370, 0.2558438, -0.1424370, 0.2558438, -0.3982807, 0.3982807)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.53 + 2.02 = 3.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.4351534, upper bound: 0.4351535

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 138
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 138

## Relational analysis of IS_A1

### Relational analysis result of IS_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4336004, upper bound: 0.4299459
time: 1.09 seconds

## Relational analysis of IS_A2

### Relational analysis result of IS_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4336589, upper bound: 0.4336589
time: 1.11 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 2.36 seconds
IS_A1, status: Status.UNKNOWN, split count: 1, time: 2.36
Output dim: 7, lower bound: -0.4336004, upper bound: 0.4299459
IS_A2, status: Status.UNKNOWN, split count: 1, time: 2.36
Output dim: 7, lower bound: -0.4336589, upper bound: 0.4336589

## BFS IS instance: IS_A1

### Backsubstitution after applying IS history:
0: -0.1960377, 0.1375937, -0.1979434, 0.1396213, -0.3356591, 0.3355370
1: -0.1824778, 0.1463837, -0.1843823, 0.1484481, -0.3309258, 0.3307660
2: -0.1201339, 0.2390679, -0.1229712, 0.2414077, -0.3615416, 0.3620391
3: -0.0981418, 0.3004766, -0.1075496, 0.3032054, -0.3961874, 0.4017645
4: -0.1585545, 0.1860206, -0.1602317, 0.1885688, -0.3471232, 0.3462524
5: -0.1477430, 0.2228673, -0.1497627, 0.2248306, -0.3725736, 0.3726299
6: -0.1849180, 0.1748001, -0.1868866, 0.1774569, -0.3623749, 0.3616866
7: 0.5418724, 1.0723369, 0.5380285, 1.0821526, -0.5402802, 0.5343084
8: -0.1348855, 0.2581588, -0.1366783, 0.2610146, -0.3959001, 0.3948371
9: -0.1388732, 0.2525599, -0.1414462, 0.2547720, -0.3936452, 0.3940061

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=33, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A1_A1

### Relational analysis result of IS_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4281386, upper bound: 0.4212007
time: 1.11 seconds

## Relational analysis of IS_A1_A2

### Relational analysis result of IS_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4244499, upper bound: 0.4210974
time: 1.02 seconds

## BFS IS instance: IS_A2

### Backsubstitution after applying IS history:
0: -0.1959853, 0.1378033, -0.1989612, 0.1405566, -0.3365420, 0.3367645
1: -0.1827565, 0.1465794, -0.1852108, 0.1494017, -0.3321581, 0.3317902
2: -0.1209272, 0.2393481, -0.1239861, 0.2424676, -0.3633948, 0.3633342
3: -0.1044368, 0.3006869, -0.1087685, 0.3044794, -0.4037369, 0.4031968
4: -0.1585966, 0.1859789, -0.1610783, 0.1899182, -0.3485147, 0.3470572
5: -0.1477731, 0.2228813, -0.1507930, 0.2258438, -0.3736169, 0.3736743
6: -0.1849551, 0.1750727, -0.1878836, 0.1786860, -0.3636411, 0.3629563
7: 0.5412580, 1.0790758, 0.5364122, 1.0834043, -0.5421463, 0.5426636
8: -0.1347447, 0.2586126, -0.1376918, 0.2622380, -0.3969827, 0.3963044
9: -0.1394769, 0.2527034, -0.1424370, 0.2558438, -0.3953207, 0.3951403

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=20, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A2_B1

### Relational analysis result of IS_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4299459, upper bound: 0.4336004
time: 1.11 seconds

## Relational analysis of IS_A2_B2

### Relational analysis result of IS_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4299459, upper bound: 0.4336589
time: 1.18 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 4.01 seconds
IS_A1_A1, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 7, lower bound: -0.4281386, upper bound: 0.4212007
IS_A1_A2, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 7, lower bound: -0.4244499, upper bound: 0.4210974
IS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 7, lower bound: -0.4299459, upper bound: 0.4336004
IS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 4.01
Output dim: 7, lower bound: -0.4299459, upper bound: 0.4336589

## BFS IS instance: IS_A1_A1

### Backsubstitution after applying IS history:
0: -0.1791032, 0.1223922, -0.1979434, 0.1396213, -0.3187246, 0.3203356
1: -0.1688330, 0.1295736, -0.1843823, 0.1484481, -0.3172811, 0.3139559
2: -0.1044585, 0.2213965, -0.1229712, 0.2414077, -0.3458662, 0.3443677
3: -0.0965597, 0.2790132, -0.1075496, 0.3032054, -0.3943235, 0.3803070
4: -0.1440112, 0.1632273, -0.1602317, 0.1885688, -0.3325800, 0.3234590
5: -0.1307711, 0.2041727, -0.1497627, 0.2248306, -0.3556017, 0.3539354
6: -0.1672360, 0.1551028, -0.1868866, 0.1774569, -0.3446929, 0.3419894
7: 0.5682518, 1.0709462, 0.5380285, 1.0821526, -0.5139008, 0.5329177
8: -0.1161288, 0.2385267, -0.1366783, 0.2610146, -0.3771433, 0.3752049
9: -0.1236560, 0.2337611, -0.1414462, 0.2547720, -0.3784280, 0.3752073

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A1_B1

### Relational analysis result of IS_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4244499, upper bound: 0.4210974
time: 1.16 seconds

## Relational analysis of IS_A1_A1_B2

### Relational analysis result of IS_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4244499, upper bound: 0.4210974
time: 1.04 seconds

## BFS IS instance: IS_A1_A2

### Backsubstitution after applying IS history:
0: -0.1804025, 0.1236696, -0.1953881, 0.1373702, -0.3177727, 0.3190577
1: -0.1701152, 0.1312663, -0.1824203, 0.1461624, -0.3162776, 0.3136865
2: -0.1062283, 0.2229695, -0.1207779, 0.2388668, -0.3450951, 0.3437474
3: -0.1012890, 0.2811121, -0.1073395, 0.3000050, -0.3961258, 0.3823358
4: -0.1452372, 0.1650544, -0.1581306, 0.1851875, -0.3304247, 0.3231850
5: -0.1321446, 0.2058423, -0.1472041, 0.2222981, -0.3544427, 0.3530464
6: -0.1688091, 0.1567670, -0.1844188, 0.1744891, -0.3432981, 0.3411858
7: 0.5654473, 1.0744244, 0.5419453, 1.0819625, -0.5165151, 0.5324790
8: -0.1176876, 0.2403003, -0.1340845, 0.2581615, -0.3758492, 0.3743848
9: -0.1252088, 0.2354038, -0.1392258, 0.2521468, -0.3773556, 0.3746296

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=15, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=19, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_A2_B1

### Relational analysis result of IS_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
time: 1.56 seconds

## Relational analysis of IS_A1_A2_B2

### Relational analysis result of IS_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
time: 0.99 seconds

## BFS IS instance: IS_A2_B1

### Backsubstitution after applying IS history:
0: -0.1959853, 0.1378033, -0.1960377, 0.1375937, -0.3335790, 0.3338411
1: -0.1827565, 0.1465794, -0.1824778, 0.1463837, -0.3291401, 0.3290572
2: -0.1209272, 0.2393481, -0.1201339, 0.2390679, -0.3599952, 0.3594820
3: -0.1044368, 0.3006869, -0.0981418, 0.3004766, -0.3986722, 0.3925228
4: -0.1585966, 0.1859789, -0.1585545, 0.1860206, -0.3446172, 0.3445334
5: -0.1477731, 0.2228813, -0.1477430, 0.2228673, -0.3706403, 0.3706243
6: -0.1849551, 0.1750727, -0.1849180, 0.1748001, -0.3597552, 0.3599907
7: 0.5412580, 1.0790758, 0.5418724, 1.0723369, -0.5310789, 0.5372034
8: -0.1347447, 0.2586126, -0.1348855, 0.2581588, -0.3929035, 0.3934981
9: -0.1394769, 0.2527034, -0.1388732, 0.2525599, -0.3920369, 0.3915766

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4212007, upper bound: 0.4281386
time: 1.11 seconds

## Relational analysis of IS_A2_B1_B2

### Relational analysis result of IS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244499
time: 1.30 seconds

## BFS IS instance: IS_A2_B2

### Backsubstitution after applying IS history:
0: -0.1959853, 0.1378033, -0.1959853, 0.1378033, -0.3337887, 0.3337887
1: -0.1827565, 0.1465794, -0.1827565, 0.1465794, -0.3293358, 0.3293358
2: -0.1209272, 0.2393481, -0.1209272, 0.2393481, -0.3602754, 0.3602754
3: -0.1044368, 0.3006869, -0.1044368, 0.3006869, -0.3987446, 0.3987445
4: -0.1585966, 0.1859789, -0.1585966, 0.1859789, -0.3445755, 0.3445755
5: -0.1477731, 0.2228813, -0.1477731, 0.2228813, -0.3706543, 0.3706543
6: -0.1849551, 0.1750727, -0.1849551, 0.1750727, -0.3600278, 0.3600278
7: 0.5412580, 1.0790758, 0.5412580, 1.0790758, -0.5378178, 0.5378178
8: -0.1347447, 0.2586126, -0.1347447, 0.2586126, -0.3933573, 0.3933573
9: -0.1394769, 0.2527034, -0.1394769, 0.2527034, -0.3921803, 0.3921803

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1

### Relational analysis result of IS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4240709, upper bound: 0.4245393
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A2

### Relational analysis result of IS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244533
time: 1.19 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 3.91 seconds
IS_A1_A1_B1, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 7, lower bound: -0.4244499, upper bound: 0.4210974
IS_A1_A1_B2, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 7, lower bound: -0.4244499, upper bound: 0.4210974
IS_A1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
IS_A1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
IS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 7, lower bound: -0.4212007, upper bound: 0.4281386
IS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244499
IS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 7, lower bound: -0.4240709, upper bound: 0.4245393
IS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.91
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244533

## BFS IS instance: IS_A1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1791032, 0.1223922, -0.1803333, 0.1239385, -0.3030418, 0.3027255
1: -0.1688330, 0.1295736, -0.1703650, 0.1313501, -0.3001831, 0.2999386
2: -0.1044585, 0.2213965, -0.1069440, 0.2233187, -0.3277772, 0.3283406
3: -0.0965597, 0.2790132, -0.1059328, 0.2809974, -0.3709174, 0.3785080
4: -0.1440112, 0.1632273, -0.1452632, 0.1650970, -0.3091083, 0.3084905
5: -0.1307711, 0.2041727, -0.1321810, 0.2058432, -0.3366143, 0.3363538
6: -0.1672360, 0.1551028, -0.1687816, 0.1571426, -0.3243786, 0.3238844
7: 0.5682518, 1.0709462, 0.5652615, 1.0807118, -0.5124600, 0.5056847
8: -0.1161288, 0.2385267, -0.1175674, 0.2407665, -0.3568953, 0.3560941
9: -0.1236560, 0.2337611, -0.1258148, 0.2355854, -0.3592414, 0.3595759

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4242924, upper bound: 0.4212007
time: 1.23 seconds

## Relational analysis of IS_A1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4242924, upper bound: 0.4212007
time: 1.11 seconds

## BFS IS instance: IS_A1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1791032, 0.1223922, -0.1818847, 0.1253517, -0.3044549, 0.3042769
1: -0.1688330, 0.1295736, -0.1718149, 0.1333731, -0.3022060, 0.3013886
2: -0.1044585, 0.2213965, -0.1089381, 0.2250595, -0.3295180, 0.3303346
3: -0.0965597, 0.2790132, -0.1107652, 0.2836398, -0.3736961, 0.3836275
4: -0.1440112, 0.1632273, -0.1466622, 0.1671999, -0.3112111, 0.3098895
5: -0.1307711, 0.2041727, -0.1337824, 0.2077431, -0.3385143, 0.3379551
6: -0.1672360, 0.1551028, -0.1706257, 0.1589896, -0.3262256, 0.3257285
7: 0.5682518, 1.0709462, 0.5617999, 1.0842134, -0.5159615, 0.5091463
8: -0.1161288, 0.2385267, -0.1193544, 0.2427973, -0.3589261, 0.3578811
9: -0.1236560, 0.2337611, -0.1275789, 0.2374181, -0.3610741, 0.3613400

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 138
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 138

## Relational analysis of IS_A1_A1_B2_B1

### Relational analysis result of IS_A1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4242924, upper bound: 0.4212007
time: 1.21 seconds

## Relational analysis of IS_A1_A1_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4242924, upper bound: 0.4212007
time: 1.30 seconds

## BFS IS instance: IS_A1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1804025, 0.1236696, -0.1934887, 0.1353241, -0.3157266, 0.3171584
1: -0.1701152, 0.1312663, -0.1805065, 0.1440809, -0.3141961, 0.3117727
2: -0.1062283, 0.2229695, -0.1179295, 0.2365082, -0.3427365, 0.3408990
3: -0.1012890, 0.2811121, -0.0979348, 0.2972408, -0.3922306, 0.3728891
4: -0.1452372, 0.1650544, -0.1564431, 0.1826147, -0.3278518, 0.3214975
5: -0.1321446, 0.2058423, -0.1451742, 0.2203127, -0.3524573, 0.3510165
6: -0.1688091, 0.1567670, -0.1824393, 0.1718106, -0.3406197, 0.3392063
7: 0.5654473, 1.0744244, 0.5458286, 1.0721530, -0.5067056, 0.5285958
8: -0.1176876, 0.2403003, -0.1322716, 0.2553072, -0.3729948, 0.3725718
9: -0.1252088, 0.2354038, -0.1366347, 0.2499173, -0.3751261, 0.3720385

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=33, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A2_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
time: 1.05 seconds

## Relational analysis of IS_A1_A2_B1_B2

### Relational analysis result of IS_A1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
time: 1.21 seconds

## BFS IS instance: IS_A1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1804025, 0.1236696, -0.1934484, 0.1355680, -0.3159704, 0.3171180
1: -0.1701152, 0.1312663, -0.1808083, 0.1443100, -0.3144252, 0.3120745
2: -0.1062283, 0.2229695, -0.1187492, 0.2368262, -0.3430545, 0.3417187
3: -0.1012890, 0.2811121, -0.1042297, 0.2975072, -0.3924716, 0.3792507
4: -0.1452372, 0.1650544, -0.1565100, 0.1826241, -0.3278613, 0.3215644
5: -0.1321446, 0.2058423, -0.1452336, 0.2203655, -0.3525101, 0.3510758
6: -0.1688091, 0.1567670, -0.1825039, 0.1721272, -0.3409363, 0.3392708
7: 0.5654473, 1.0744244, 0.5451462, 1.0788875, -0.5134401, 0.5292782
8: -0.1176876, 0.2403003, -0.1321702, 0.2557800, -0.3734676, 0.3724704
9: -0.1252088, 0.2354038, -0.1372723, 0.2500978, -0.3753066, 0.3726761

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A1_A2_B2_B1

### Relational analysis result of IS_A1_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
time: 1.27 seconds

## Relational analysis of IS_A1_A2_B2_B2

### Relational analysis result of IS_A1_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1959853, 0.1378033, -0.1791032, 0.1223922, -0.3183776, 0.3169065
1: -0.1827565, 0.1465794, -0.1688330, 0.1295736, -0.3123301, 0.3154123
2: -0.1209272, 0.2393481, -0.1044585, 0.2213965, -0.3423238, 0.3438066
3: -0.1044368, 0.3006869, -0.0965597, 0.2790132, -0.3772148, 0.3906743
4: -0.1585966, 0.1859789, -0.1440112, 0.1632273, -0.3218238, 0.3299901
5: -0.1477731, 0.2228813, -0.1307711, 0.2041727, -0.3519458, 0.3536524
6: -0.1849551, 0.1750727, -0.1672360, 0.1551028, -0.3400579, 0.3423087
7: 0.5412580, 1.0790758, 0.5682518, 1.0709462, -0.5296882, 0.5108240
8: -0.1347447, 0.2586126, -0.1161288, 0.2385267, -0.3732714, 0.3747413
9: -0.1394769, 0.2527034, -0.1236560, 0.2337611, -0.3732381, 0.3763593

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244499
time: 1.05 seconds

## Relational analysis of IS_A2_B1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244499
time: 1.06 seconds

## BFS IS instance: IS_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1934484, 0.1355680, -0.1804025, 0.1236696, -0.3171180, 0.3159704
1: -0.1808083, 0.1443100, -0.1701152, 0.1312663, -0.3120745, 0.3144252
2: -0.1187492, 0.2368262, -0.1062283, 0.2229695, -0.3417187, 0.3430545
3: -0.1042297, 0.2975072, -0.1012890, 0.2811121, -0.3792507, 0.3924715
4: -0.1565100, 0.1826241, -0.1452372, 0.1650544, -0.3215644, 0.3278613
5: -0.1452336, 0.2203655, -0.1321446, 0.2058423, -0.3510758, 0.3525101
6: -0.1825039, 0.1721272, -0.1688091, 0.1567670, -0.3392708, 0.3409363
7: 0.5451462, 1.0788875, 0.5654473, 1.0744244, -0.5292782, 0.5134401
8: -0.1321702, 0.2557800, -0.1176876, 0.2403003, -0.3724704, 0.3734676
9: -0.1372723, 0.2500978, -0.1252088, 0.2354038, -0.3726761, 0.3753066

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=14, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 85

## Relational analysis of IS_A2_B1_B2_A1

### Relational analysis result of IS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244499
time: 1.07 seconds

## Relational analysis of IS_A2_B1_B2_A2

### Relational analysis result of IS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244499
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1788291, 0.1224563, -0.1959853, 0.1378033, -0.3166324, 0.3184417
1: -0.1689402, 0.1295181, -0.1827565, 0.1465794, -0.3155196, 0.3122746
2: -0.1050421, 0.2215055, -0.1209272, 0.2393481, -0.3443902, 0.3424327
3: -0.1028352, 0.2789302, -0.1044368, 0.3006869, -0.3969666, 0.3769419
4: -0.1438657, 0.1629859, -0.1585966, 0.1859789, -0.3298446, 0.3215825
5: -0.1306279, 0.2039206, -0.1477731, 0.2228813, -0.3535091, 0.3516936
6: -0.1670104, 0.1552319, -0.1849551, 0.1750727, -0.3420831, 0.3401870
7: 0.5679736, 1.0776608, 0.5412580, 1.0790758, -0.5111022, 0.5364028
8: -0.1157474, 0.2387219, -0.1347447, 0.2586126, -0.3743600, 0.3734666
9: -0.1241060, 0.2336740, -0.1394769, 0.2527034, -0.3768094, 0.3731509

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4218049, upper bound: 0.4244533
time: 1.09 seconds

## Relational analysis of IS_A2_B2_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4218049, upper bound: 0.4244533
time: 1.03 seconds

## BFS IS instance: IS_A2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1803547, 0.1239133, -0.1934484, 0.1355680, -0.3159227, 0.3173617
1: -0.1704204, 0.1316042, -0.1808083, 0.1443100, -0.3147304, 0.3124124
2: -0.1070789, 0.2233019, -0.1187492, 0.2368262, -0.3439051, 0.3420511
3: -0.1077114, 0.2816291, -0.1042297, 0.2975072, -0.3989488, 0.3795888
4: -0.1452897, 0.1651578, -0.1565100, 0.1826241, -0.3279138, 0.3216678
5: -0.1322553, 0.2058822, -0.1452336, 0.2203655, -0.3526208, 0.3511157
6: -0.1688845, 0.1571378, -0.1825039, 0.1721272, -0.3410117, 0.3396417
7: 0.5644401, 1.0811839, 0.5451462, 1.0788875, -0.5144473, 0.5360377
8: -0.1175945, 0.2407477, -0.1321702, 0.2557800, -0.3733745, 0.3729180
9: -0.1259222, 0.2355606, -0.1372723, 0.2500978, -0.3760200, 0.3728328

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=14, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=34, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 85

## Relational analysis of IS_A2_B2_A2_B1

### Relational analysis result of IS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4218049, upper bound: 0.4244533
time: 1.00 seconds

## Relational analysis of IS_A2_B2_A2_B2

### Relational analysis result of IS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4218049, upper bound: 0.4244533
time: 1.02 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 3.63 seconds
IS_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4242924, upper bound: 0.4212007
IS_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4242924, upper bound: 0.4212007
IS_A1_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4242924, upper bound: 0.4212007
IS_A1_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4242924, upper bound: 0.4212007
IS_A1_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
IS_A1_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
IS_A1_A2_B2_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
IS_A1_A2_B2_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4211127, upper bound: 0.4210974
IS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244499
IS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244499
IS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244499
IS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4210974, upper bound: 0.4244499
IS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4218049, upper bound: 0.4244533
IS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4218049, upper bound: 0.4244533
IS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4218049, upper bound: 0.4244533
IS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.63
Output dim: 7, lower bound: -0.4218049, upper bound: 0.4244533

## BFS IS instance: IS_A1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1791032, 0.1223922, -0.1791032, 0.1223922, -0.3014954, 0.3014954
1: -0.1688330, 0.1295736, -0.1688330, 0.1295736, -0.2984066, 0.2984066
2: -0.1044585, 0.2213965, -0.1044585, 0.2213965, -0.3258551, 0.3258551
3: -0.0965597, 0.2790132, -0.0965597, 0.2790132, -0.3690261, 0.3690261
4: -0.1440112, 0.1632273, -0.1440112, 0.1632273, -0.3072385, 0.3072385
5: -0.1307711, 0.2041727, -0.1307711, 0.2041727, -0.3349438, 0.3349438
6: -0.1672360, 0.1551028, -0.1672360, 0.1551028, -0.3223388, 0.3223388
7: 0.5682518, 1.0709462, 0.5682518, 1.0709462, -0.5026944, 0.5026944
8: -0.1161288, 0.2385267, -0.1161288, 0.2385267, -0.3546554, 0.3546554
9: -0.1236560, 0.2337611, -0.1236560, 0.2337611, -0.3574171, 0.3574171

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4055543, upper bound: 0.3783335
time: 0.99 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_B1_B2

### Backsubstitution after applying IS history:
0: -0.1791032, 0.1223922, -0.1788291, 0.1224563, -0.3015595, 0.3012213
1: -0.1688330, 0.1295736, -0.1689402, 0.1295181, -0.2983511, 0.2985138
2: -0.1044585, 0.2213965, -0.1050421, 0.2215055, -0.3259640, 0.3264386
3: -0.0965597, 0.2790132, -0.1028352, 0.2789302, -0.3688685, 0.3754346
4: -0.1440112, 0.1632273, -0.1438657, 0.1629859, -0.3069972, 0.3070930
5: -0.1307711, 0.2041727, -0.1306279, 0.2039206, -0.3346917, 0.3348006
6: -0.1672360, 0.1551028, -0.1670104, 0.1552319, -0.3224679, 0.3221132
7: 0.5682518, 1.0709462, 0.5679736, 1.0776608, -0.5094090, 0.5029726
8: -0.1161288, 0.2385267, -0.1157474, 0.2387219, -0.3548507, 0.3542741
9: -0.1236560, 0.2337611, -0.1241060, 0.2336740, -0.3573300, 0.3578672

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4055543, upper bound: 0.3783395
time: 0.96 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
time: 0.98 seconds

## BFS IS instance: IS_A1_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1791032, 0.1223922, -0.1804025, 0.1236696, -0.3027728, 0.3027947
1: -0.1688330, 0.1295736, -0.1701152, 0.1312663, -0.3000993, 0.2996888
2: -0.1044585, 0.2213965, -0.1062283, 0.2229695, -0.3274280, 0.3276248
3: -0.0965597, 0.2790132, -0.1012890, 0.2811121, -0.3712326, 0.3739859
4: -0.1440112, 0.1632273, -0.1452372, 0.1650544, -0.3090656, 0.3084645
5: -0.1307711, 0.2041727, -0.1321446, 0.2058423, -0.3366134, 0.3363173
6: -0.1672360, 0.1551028, -0.1688091, 0.1567670, -0.3240030, 0.3239119
7: 0.5682518, 1.0709462, 0.5654473, 1.0744244, -0.5061725, 0.5054989
8: -0.1161288, 0.2385267, -0.1176876, 0.2403003, -0.3564290, 0.3562143
9: -0.1236560, 0.2337611, -0.1252088, 0.2354038, -0.3590598, 0.3589699

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B2_B1_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3761118, upper bound: 0.4004246
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2

### Relational analysis result of IS_A1_A1_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3695155
time: 0.78 seconds

## BFS IS instance: IS_A1_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1791032, 0.1223922, -0.1803547, 0.1239133, -0.3030165, 0.3027469
1: -0.1688330, 0.1295736, -0.1704204, 0.1316042, -0.3004372, 0.2999940
2: -0.1044585, 0.2213965, -0.1070789, 0.2233019, -0.3277604, 0.3284754
3: -0.0965597, 0.2790132, -0.1077114, 0.2816291, -0.3717024, 0.3805739
4: -0.1440112, 0.1632273, -0.1452897, 0.1651578, -0.3091690, 0.3085170
5: -0.1307711, 0.2041727, -0.1322553, 0.2058822, -0.3366533, 0.3364280
6: -0.1672360, 0.1551028, -0.1688845, 0.1571378, -0.3243738, 0.3239873
7: 0.5682518, 1.0709462, 0.5644401, 1.0811839, -0.5129321, 0.5065061
8: -0.1161288, 0.2385267, -0.1175945, 0.2407477, -0.3568765, 0.3561212
9: -0.1236560, 0.2337611, -0.1259222, 0.2355606, -0.3592166, 0.3596833

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.57 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B2_B2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3761118, upper bound: 0.4004246
time: 1.04 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3697596
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1804025, 0.1236696, -0.1791032, 0.1223922, -0.3027947, 0.3027728
1: -0.1701152, 0.1312663, -0.1688330, 0.1295736, -0.2996888, 0.3000993
2: -0.1062283, 0.2229695, -0.1044585, 0.2213965, -0.3276248, 0.3274280
3: -0.1012890, 0.2811121, -0.0965597, 0.2790132, -0.3739859, 0.3712326
4: -0.1452372, 0.1650544, -0.1440112, 0.1632273, -0.3084645, 0.3090656
5: -0.1321446, 0.2058423, -0.1307711, 0.2041727, -0.3363173, 0.3366134
6: -0.1688091, 0.1567670, -0.1672360, 0.1551028, -0.3239119, 0.3240030
7: 0.5654473, 1.0744244, 0.5682518, 1.0709462, -0.5054989, 0.5061725
8: -0.1176876, 0.2403003, -0.1161288, 0.2385267, -0.3562143, 0.3564290
9: -0.1252088, 0.2354038, -0.1236560, 0.2337611, -0.3589699, 0.3590598

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B1_B1_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4004294, upper bound: 0.3713449
time: 0.97 seconds

## Relational analysis of IS_A1_A2_B1_B1_A2

### Relational analysis result of IS_A1_A2_B1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3694593, upper bound: 0.3694593
time: 0.90 seconds

## BFS IS instance: IS_A1_A2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1804025, 0.1236696, -0.1804025, 0.1236696, -0.3040721, 0.3040721
1: -0.1701152, 0.1312663, -0.1701152, 0.1312663, -0.3013814, 0.3013814
2: -0.1062283, 0.2229695, -0.1062283, 0.2229695, -0.3291978, 0.3291978
3: -0.1012890, 0.2811121, -0.1012890, 0.2811121, -0.3760753, 0.3760754
4: -0.1452372, 0.1650544, -0.1452372, 0.1650544, -0.3102916, 0.3102916
5: -0.1321446, 0.2058423, -0.1321446, 0.2058423, -0.3379869, 0.3379869
6: -0.1688091, 0.1567670, -0.1688091, 0.1567670, -0.3255761, 0.3255761
7: 0.5654473, 1.0744244, 0.5654473, 1.0744244, -0.5089771, 0.5089771
8: -0.1176876, 0.2403003, -0.1176876, 0.2403003, -0.3579879, 0.3579879
9: -0.1252088, 0.2354038, -0.1252088, 0.2354038, -0.3606126, 0.3606126

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B1_B2_B1

### Relational analysis result of IS_A1_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3713449, upper bound: 0.4004294
time: 1.00 seconds

## Relational analysis of IS_A1_A2_B1_B2_B2

### Relational analysis result of IS_A1_A2_B1_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3694593, upper bound: 0.3694593
time: 0.92 seconds

## BFS IS instance: IS_A1_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1804025, 0.1236696, -0.1788291, 0.1224563, -0.3028588, 0.3024988
1: -0.1701152, 0.1312663, -0.1689402, 0.1295181, -0.2996333, 0.3002065
2: -0.1062283, 0.2229695, -0.1050421, 0.2215055, -0.3277338, 0.3280115
3: -0.1012890, 0.2811121, -0.1028352, 0.2789302, -0.3738284, 0.3776411
4: -0.1452372, 0.1650544, -0.1438657, 0.1629859, -0.3082231, 0.3089201
5: -0.1321446, 0.2058423, -0.1306279, 0.2039206, -0.3360652, 0.3364701
6: -0.1688091, 0.1567670, -0.1670104, 0.1552319, -0.3240410, 0.3237773
7: 0.5654473, 1.0744244, 0.5679736, 1.0776608, -0.5122135, 0.5064508
8: -0.1176876, 0.2403003, -0.1157474, 0.2387219, -0.3564096, 0.3560477
9: -0.1252088, 0.2354038, -0.1241060, 0.2336740, -0.3588828, 0.3595098

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B2_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4029411, upper bound: 0.3715217
time: 0.98 seconds

## Relational analysis of IS_A1_A2_B2_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3712151, upper bound: 0.3697016
time: 1.01 seconds

## BFS IS instance: IS_A1_A2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1804025, 0.1236696, -0.1803547, 0.1239133, -0.3043157, 0.3040243
1: -0.1701152, 0.1312663, -0.1704204, 0.1316042, -0.3017194, 0.3016866
2: -0.1062283, 0.2229695, -0.1070789, 0.2233019, -0.3295302, 0.3300483
3: -0.1012890, 0.2811121, -0.1077114, 0.2816291, -0.3765332, 0.3826742
4: -0.1452372, 0.1650544, -0.1452897, 0.1651578, -0.3103949, 0.3103441
5: -0.1321446, 0.2058423, -0.1322553, 0.2058822, -0.3380268, 0.3380976
6: -0.1688091, 0.1567670, -0.1688845, 0.1571378, -0.3259469, 0.3256515
7: 0.5654473, 1.0744244, 0.5644401, 1.0811839, -0.5157366, 0.5099843
8: -0.1176876, 0.2403003, -0.1175945, 0.2407477, -0.3584354, 0.3578948
9: -0.1252088, 0.2354038, -0.1259222, 0.2355606, -0.3607694, 0.3613260

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B2_B2_A1

### Relational analysis result of IS_A1_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4029411, upper bound: 0.3715217
time: 1.04 seconds

## Relational analysis of IS_A1_A2_B2_B2_A2

### Relational analysis result of IS_A1_A2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3712151, upper bound: 0.3697016
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1788291, 0.1224563, -0.1791032, 0.1223922, -0.3012213, 0.3015595
1: -0.1689402, 0.1295181, -0.1688330, 0.1295736, -0.2985138, 0.2983511
2: -0.1050421, 0.2215055, -0.1044585, 0.2213965, -0.3264386, 0.3259640
3: -0.1028352, 0.2789302, -0.0965597, 0.2790132, -0.3754346, 0.3688685
4: -0.1438657, 0.1629859, -0.1440112, 0.1632273, -0.3070930, 0.3069972
5: -0.1306279, 0.2039206, -0.1307711, 0.2041727, -0.3348006, 0.3346917
6: -0.1670104, 0.1552319, -0.1672360, 0.1551028, -0.3221132, 0.3224679
7: 0.5679736, 1.0776608, 0.5682518, 1.0709462, -0.5029726, 0.5094090
8: -0.1157474, 0.2387219, -0.1161288, 0.2385267, -0.3542741, 0.3548507
9: -0.1241060, 0.2336740, -0.1236560, 0.2337611, -0.3578672, 0.3573300

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3719053, upper bound: 0.4058383
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3697596, upper bound: 0.3765575
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1803547, 0.1239133, -0.1791032, 0.1223922, -0.3027469, 0.3030165
1: -0.1704204, 0.1316042, -0.1688330, 0.1295736, -0.2999940, 0.3004372
2: -0.1070789, 0.2233019, -0.1044585, 0.2213965, -0.3284754, 0.3277604
3: -0.1077114, 0.2816291, -0.0965597, 0.2790132, -0.3805739, 0.3717024
4: -0.1452897, 0.1651578, -0.1440112, 0.1632273, -0.3085170, 0.3091690
5: -0.1322553, 0.2058822, -0.1307711, 0.2041727, -0.3364280, 0.3366533
6: -0.1688845, 0.1571378, -0.1672360, 0.1551028, -0.3239873, 0.3243738
7: 0.5644401, 1.0811839, 0.5682518, 1.0709462, -0.5065061, 0.5129321
8: -0.1175945, 0.2407477, -0.1161288, 0.2385267, -0.3561212, 0.3568765
9: -0.1259222, 0.2355606, -0.1236560, 0.2337611, -0.3596833, 0.3592166

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_B1_A2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4004246, upper bound: 0.3786117
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3697596, upper bound: 0.3765575
time: 0.90 seconds

## BFS IS instance: IS_A2_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1788291, 0.1224563, -0.1804025, 0.1236696, -0.3024988, 0.3028588
1: -0.1689402, 0.1295181, -0.1701152, 0.1312663, -0.3002065, 0.2996333
2: -0.1050421, 0.2215055, -0.1062283, 0.2229695, -0.3280115, 0.3277338
3: -0.1028352, 0.2789302, -0.1012890, 0.2811121, -0.3776411, 0.3738284
4: -0.1438657, 0.1629859, -0.1452372, 0.1650544, -0.3089201, 0.3082231
5: -0.1306279, 0.2039206, -0.1321446, 0.2058423, -0.3364701, 0.3360652
6: -0.1670104, 0.1552319, -0.1688091, 0.1567670, -0.3237773, 0.3240410
7: 0.5679736, 1.0776608, 0.5654473, 1.0744244, -0.5064508, 0.5122135
8: -0.1157474, 0.2387219, -0.1176876, 0.2403003, -0.3560477, 0.3564096
9: -0.1241060, 0.2336740, -0.1252088, 0.2354038, -0.3595098, 0.3588828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_B2_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3715217, upper bound: 0.4029411
time: 1.01 seconds

## Relational analysis of IS_A2_B1_B2_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3697016, upper bound: 0.3712151
time: 0.98 seconds

## BFS IS instance: IS_A2_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1803547, 0.1239133, -0.1804025, 0.1236696, -0.3040243, 0.3043157
1: -0.1704204, 0.1316042, -0.1701152, 0.1312663, -0.3016866, 0.3017194
2: -0.1070789, 0.2233019, -0.1062283, 0.2229695, -0.3300483, 0.3295302
3: -0.1077114, 0.2816291, -0.1012890, 0.2811121, -0.3826743, 0.3765332
4: -0.1452897, 0.1651578, -0.1452372, 0.1650544, -0.3103441, 0.3103949
5: -0.1322553, 0.2058822, -0.1321446, 0.2058423, -0.3380976, 0.3380268
6: -0.1688845, 0.1571378, -0.1688091, 0.1567670, -0.3256515, 0.3259469
7: 0.5644401, 1.0811839, 0.5654473, 1.0744244, -0.5099843, 0.5157366
8: -0.1175945, 0.2407477, -0.1176876, 0.2403003, -0.3578948, 0.3584354
9: -0.1259222, 0.2355606, -0.1252088, 0.2354038, -0.3613260, 0.3607694

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3715217, upper bound: 0.4029445
time: 1.03 seconds

## Relational analysis of IS_A2_B1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3697016, upper bound: 0.3712151
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1788291, 0.1224563, -0.1788291, 0.1224563, -0.3012854, 0.3012854
1: -0.1689402, 0.1295181, -0.1689402, 0.1295181, -0.2984583, 0.2984583
2: -0.1050421, 0.2215055, -0.1050421, 0.2215055, -0.3265476, 0.3265476
3: -0.1028352, 0.2789302, -0.1028352, 0.2789302, -0.3751640, 0.3751640
4: -0.1438657, 0.1629859, -0.1438657, 0.1629859, -0.3068516, 0.3068516
5: -0.1306279, 0.2039206, -0.1306279, 0.2039206, -0.3345484, 0.3345484
6: -0.1670104, 0.1552319, -0.1670104, 0.1552319, -0.3222423, 0.3222423
7: 0.5679736, 1.0776608, 0.5679736, 1.0776608, -0.5096872, 0.5096872
8: -0.1157474, 0.2387219, -0.1157474, 0.2387219, -0.3544693, 0.3544693
9: -0.1241060, 0.2336740, -0.1241060, 0.2336740, -0.3577800, 0.3577800

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4041851, upper bound: 0.3747851
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
time: 0.93 seconds

## BFS IS instance: IS_A2_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1788291, 0.1224563, -0.1803547, 0.1239133, -0.3027424, 0.3028110
1: -0.1689402, 0.1295181, -0.1704204, 0.1316042, -0.3005444, 0.2999385
2: -0.1050421, 0.2215055, -0.1070789, 0.2233019, -0.3283440, 0.3285844
3: -0.1028352, 0.2789302, -0.1077114, 0.2816291, -0.3779950, 0.3803092
4: -0.1438657, 0.1629859, -0.1452897, 0.1651578, -0.3090234, 0.3082756
5: -0.1306279, 0.2039206, -0.1322553, 0.2058822, -0.3365100, 0.3361759
6: -0.1670104, 0.1552319, -0.1688845, 0.1571378, -0.3241482, 0.3241164
7: 0.5679736, 1.0776608, 0.5644401, 1.0811839, -0.5132103, 0.5132207
8: -0.1157474, 0.2387219, -0.1175945, 0.2407477, -0.3564951, 0.3563164
9: -0.1241060, 0.2336740, -0.1259222, 0.2355606, -0.3596666, 0.3595962

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=32, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A1_B2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3771509, upper bound: 0.4030094
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
time: 0.94 seconds

## BFS IS instance: IS_A2_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1803547, 0.1239133, -0.1788291, 0.1224563, -0.3028110, 0.3027424
1: -0.1704204, 0.1316042, -0.1689402, 0.1295181, -0.2999385, 0.3005444
2: -0.1070789, 0.2233019, -0.1050421, 0.2215055, -0.3285844, 0.3283440
3: -0.1077114, 0.2816291, -0.1028352, 0.2789302, -0.3803092, 0.3779950
4: -0.1452897, 0.1651578, -0.1438657, 0.1629859, -0.3082756, 0.3090234
5: -0.1322553, 0.2058822, -0.1306279, 0.2039206, -0.3361759, 0.3365100
6: -0.1688845, 0.1571378, -0.1670104, 0.1552319, -0.3241164, 0.3241482
7: 0.5644401, 1.0811839, 0.5679736, 1.0776608, -0.5132207, 0.5132103
8: -0.1175945, 0.2407477, -0.1157474, 0.2387219, -0.3563164, 0.3564951
9: -0.1259222, 0.2355606, -0.1241060, 0.2336740, -0.3595962, 0.3596666

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4014348, upper bound: 0.3747399
time: 0.97 seconds

## Relational analysis of IS_A2_B2_A2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3713892, upper bound: 0.3723255
time: 1.01 seconds

## BFS IS instance: IS_A2_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1803547, 0.1239133, -0.1803547, 0.1239133, -0.3042680, 0.3042680
1: -0.1704204, 0.1316042, -0.1704204, 0.1316042, -0.3020245, 0.3020245
2: -0.1070789, 0.2233019, -0.1070789, 0.2233019, -0.3303807, 0.3303807
3: -0.1077114, 0.2816291, -0.1077114, 0.2816291, -0.3830193, 0.3830193
4: -0.1452897, 0.1651578, -0.1452897, 0.1651578, -0.3104475, 0.3104475
5: -0.1322553, 0.2058822, -0.1322553, 0.2058822, -0.3381375, 0.3381375
6: -0.1688845, 0.1571378, -0.1688845, 0.1571378, -0.3260223, 0.3260223
7: 0.5644401, 1.0811839, 0.5644401, 1.0811839, -0.5167438, 0.5167438
8: -0.1175945, 0.2407477, -0.1175945, 0.2407477, -0.3583422, 0.3583422
9: -0.1259222, 0.2355606, -0.1259222, 0.2355606, -0.3614828, 0.3614828

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.42 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A2_B2_B1

### Relational analysis result of IS_A2_B2_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3732577, upper bound: 0.4030094
time: 2.03 seconds

## Relational analysis of IS_A2_B2_A2_B2_B2

### Relational analysis result of IS_A2_B2_A2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3713892, upper bound: 0.3723255
time: 1.16 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 4.77 seconds
IS_A1_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.4055543, upper bound: 0.3783335
IS_A1_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
IS_A1_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.4055543, upper bound: 0.3783395
IS_A1_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
IS_A1_A1_B2_B1_B1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3761118, upper bound: 0.4004246
IS_A1_A1_B2_B1_B2, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3695155
IS_A1_A1_B2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3761118, upper bound: 0.4004246
IS_A1_A1_B2_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3697596
IS_A1_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.4004294, upper bound: 0.3713449
IS_A1_A2_B1_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3694593, upper bound: 0.3694593
IS_A1_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3713449, upper bound: 0.4004294
IS_A1_A2_B1_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3694593, upper bound: 0.3694593
IS_A1_A2_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.4029411, upper bound: 0.3715217
IS_A1_A2_B2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3712151, upper bound: 0.3697016
IS_A1_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.4029411, upper bound: 0.3715217
IS_A1_A2_B2_B2_A2, status: Status.VERIFIED, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3712151, upper bound: 0.3697016
IS_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3719053, upper bound: 0.4058383
IS_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3697596, upper bound: 0.3765575
IS_A2_B1_B1_A2_A1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.4004246, upper bound: 0.3786117
IS_A2_B1_B1_A2_A2, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3697596, upper bound: 0.3765575
IS_A2_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3715217, upper bound: 0.4029411
IS_A2_B1_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3697016, upper bound: 0.3712151
IS_A2_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3715217, upper bound: 0.4029445
IS_A2_B1_B2_A2_B2, status: Status.VERIFIED, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3697016, upper bound: 0.3712151
IS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.4041851, upper bound: 0.3747851
IS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
IS_A2_B2_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3771509, upper bound: 0.4030094
IS_A2_B2_A1_B2_B2, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
IS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.4014348, upper bound: 0.3747399
IS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3713892, upper bound: 0.3723255
IS_A2_B2_A2_B2_B1, status: Status.UNKNOWN, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3732577, upper bound: 0.4030094
IS_A2_B2_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 4.77
Output dim: 7, lower bound: -0.3713892, upper bound: 0.3723255

## BFS IS instance: IS_A1_A1_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1741697, 0.1182501, -0.1791032, 0.1223922, -0.2965619, 0.2973534
1: -0.1643215, 0.1231771, -0.1688330, 0.1295736, -0.2938951, 0.2920101
2: -0.0990503, 0.2159769, -0.1044585, 0.2213965, -0.3204468, 0.3204354
3: -0.0939293, 0.2683803, -0.0965597, 0.2790132, -0.3663704, 0.3584245
4: -0.1396105, 0.1573178, -0.1440112, 0.1632273, -0.3028378, 0.3013290
5: -0.1255827, 0.1981847, -0.1307711, 0.2041727, -0.3297554, 0.3289558
6: -0.1618320, 0.1492541, -0.1672360, 0.1551028, -0.3169348, 0.3164902
7: 0.5814726, 1.0688863, 0.5682518, 1.0709462, -0.4894736, 0.5006344
8: -0.1105620, 0.2326764, -0.1161288, 0.2385267, -0.3490887, 0.3488052
9: -0.1187423, 0.2279480, -0.1236560, 0.2337611, -0.3525034, 0.3516040

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3767146, upper bound: 0.3445214
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
time: 0.94 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
time: 0.93 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1674655, 0.1138410, -0.1760962, 0.1198711, -0.2873366, 0.2899371
1: -0.1575197, 0.1119215, -0.1660738, 0.1256117, -0.2831314, 0.2779953
2: -0.0910432, 0.2086956, -0.1011454, 0.2180896, -0.3091328, 0.3098411
3: -0.0890170, 0.2423350, -0.0949562, 0.2723634, -0.3540168, 0.3315698
4: -0.1337435, 0.1498526, -0.1413345, 0.1593573, -0.2931008, 0.2911871
5: -0.1177301, 0.1908531, -0.1275935, 0.2005221, -0.3182522, 0.3184466
6: -0.1544752, 0.1411408, -0.1638376, 0.1515301, -0.3060054, 0.3049784
7: 0.6142168, 1.0700700, 0.5765510, 1.0696836, -0.4554667, 0.4935191
8: -0.1036088, 0.2256446, -0.1126227, 0.2349626, -0.3385714, 0.3382673
9: -0.1114052, 0.2205204, -0.1206113, 0.2302211, -0.3416263, 0.3411317

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
time: 0.98 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1741697, 0.1182501, -0.1788291, 0.1224563, -0.2966260, 0.2970793
1: -0.1643215, 0.1231771, -0.1689402, 0.1295181, -0.2938396, 0.2921174
2: -0.0990503, 0.2159769, -0.1050421, 0.2215055, -0.3205557, 0.3210190
3: -0.0939293, 0.2683803, -0.1028352, 0.2789302, -0.3662128, 0.3648330
4: -0.1396105, 0.1573178, -0.1438657, 0.1629859, -0.3025964, 0.3011834
5: -0.1255827, 0.1981847, -0.1306279, 0.2039206, -0.3295032, 0.3288125
6: -0.1618320, 0.1492541, -0.1670104, 0.1552319, -0.3170639, 0.3162645
7: 0.5814726, 1.0688863, 0.5679736, 1.0776608, -0.4961882, 0.5009127
8: -0.1105620, 0.2326764, -0.1157474, 0.2387219, -0.3492839, 0.3484238
9: -0.1187423, 0.2279480, -0.1241060, 0.2336740, -0.3524162, 0.3520540

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3792760, upper bound: 0.3445214
time: 0.98 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1_B2_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3781226, upper bound: 0.3763316
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3781226, upper bound: 0.3763316
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1674655, 0.1138410, -0.1757925, 0.1198970, -0.2873625, 0.2896335
1: -0.1575197, 0.1119215, -0.1661402, 0.1255416, -0.2830613, 0.2780617
2: -0.0910432, 0.2086956, -0.1016698, 0.2181470, -0.3091902, 0.3103655
3: -0.0890170, 0.2423350, -0.1012437, 0.2722510, -0.3538374, 0.3379869
4: -0.1337435, 0.1498526, -0.1411566, 0.1592279, -0.2929714, 0.2910091
5: -0.1177301, 0.1908531, -0.1274135, 0.2002448, -0.3179749, 0.3182665
6: -0.1544752, 0.1411408, -0.1636422, 0.1516037, -0.3060790, 0.3047830
7: 0.6142168, 1.0700700, 0.5763407, 1.0763991, -0.4621823, 0.4937293
8: -0.1036088, 0.2256446, -0.1122721, 0.2351013, -0.3387101, 0.3379167
9: -0.1114052, 0.2205204, -0.1210411, 0.2300825, -0.3414877, 0.3415616

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B1_B2_A2_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3781226, upper bound: 0.3763316
time: 0.98 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3781226, upper bound: 0.3763316
time: 0.97 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1

### Backsubstitution after applying IS history:
0: -0.1791032, 0.1223922, -0.1755940, 0.1196325, -0.2987357, 0.2979862
1: -0.1688330, 0.1295736, -0.1657065, 0.1249242, -0.2937571, 0.2952801
2: -0.1044585, 0.2213965, -0.1009169, 0.2176833, -0.3221418, 0.3223134
3: -0.0965597, 0.2790132, -0.0986617, 0.2706010, -0.3607631, 0.3713313
4: -0.1440112, 0.1632273, -0.1409538, 0.1588596, -0.3028709, 0.3041810
5: -0.1307711, 0.2041727, -0.1270662, 0.1999935, -0.3307646, 0.3312389
6: -0.1672360, 0.1551028, -0.1633671, 0.1510580, -0.3182940, 0.3184699
7: 0.5682518, 1.0709462, 0.5785890, 1.0723939, -0.5041420, 0.4923572
8: -0.1161288, 0.2385267, -0.1120738, 0.2345980, -0.3507268, 0.3506004
9: -0.1236560, 0.2337611, -0.1203275, 0.2297447, -0.3534007, 0.3540887

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3349383, upper bound: 0.3571597
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B2_B1_B1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3695155
time: 1.02 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3695155
time: 0.82 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B2

### Backsubstitution after applying IS history:
0: -0.1760962, 0.1198711, -0.1685045, 0.1148165, -0.2909127, 0.2883756
1: -0.1660738, 0.1256117, -0.1585342, 0.1133619, -0.2794358, 0.2841458
2: -0.1011454, 0.2180896, -0.0924540, 0.2099563, -0.3111017, 0.3105435
3: -0.0949562, 0.2723634, -0.0936355, 0.2443809, -0.3338657, 0.3588520
4: -0.1413345, 0.1593573, -0.1347321, 0.1510102, -0.2923447, 0.2940894
5: -0.1275935, 0.2005221, -0.1188443, 0.1921645, -0.3197580, 0.3193664
6: -0.1638376, 0.1515301, -0.1556403, 0.1424635, -0.3063011, 0.3071704
7: 0.5765510, 1.0696836, 0.6115620, 1.0735693, -0.4970183, 0.4581215
8: -0.1126227, 0.2349626, -0.1047032, 0.2270229, -0.3396456, 0.3396658
9: -0.1206113, 0.2302211, -0.1126498, 0.2218182, -0.3424295, 0.3428708

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B2_A1

### Relational analysis result of IS_A1_A1_B2_B1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3456531, upper bound: 0.3263008
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B2_B1_B2_A2

### Relational analysis result of IS_A1_A1_B2_B1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3211644, upper bound: 0.3182779
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1791032, 0.1223922, -0.1754827, 0.1197965, -0.2988997, 0.2978749
1: -0.1688330, 0.1295736, -0.1659405, 0.1252145, -0.2940475, 0.2955141
2: -0.1044585, 0.2213965, -0.1016621, 0.2179262, -0.3223848, 0.3230587
3: -0.0965597, 0.2790132, -0.1051774, 0.2711130, -0.3612322, 0.3771473
4: -0.1440112, 0.1632273, -0.1409440, 0.1589774, -0.3029887, 0.3041713
5: -0.1307711, 0.2041727, -0.1271106, 0.1999557, -0.3307269, 0.3312833
6: -0.1672360, 0.1551028, -0.1634101, 0.1513354, -0.3185714, 0.3185129
7: 0.5682518, 1.0709462, 0.5776169, 1.0792078, -0.5109559, 0.4933293
8: -0.1161288, 0.2385267, -0.1119504, 0.2349345, -0.3510633, 0.3504770
9: -0.1236560, 0.2337611, -0.1209784, 0.2298051, -0.3534610, 0.3547395

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3396637, upper bound: 0.3574245
time: 0.94 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A1_B2_B2_B1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3765575, upper bound: 0.3697596
time: 1.13 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3765575, upper bound: 0.3697596
time: 1.03 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B2

### Backsubstitution after applying IS history:
0: -0.1760962, 0.1198711, -0.1684408, 0.1149896, -0.2910858, 0.2883119
1: -0.1660738, 0.1256117, -0.1586944, 0.1134470, -0.2795208, 0.2843060
2: -0.1011454, 0.2180896, -0.0929544, 0.2101773, -0.3113228, 0.3110439
3: -0.0949562, 0.2723634, -0.1004950, 0.2444159, -0.3340673, 0.3658531
4: -0.1413345, 0.1593573, -0.1347493, 0.1510583, -0.2923928, 0.2941066
5: -0.1275935, 0.2005221, -0.1188599, 0.1921550, -0.3197485, 0.3193820
6: -0.1638376, 0.1515301, -0.1556504, 0.1426955, -0.3065331, 0.3071805
7: 0.5765510, 1.0696836, 0.6113134, 1.0804781, -0.5039271, 0.4583701
8: -0.1126227, 0.2349626, -0.1046044, 0.2273157, -0.3399384, 0.3395670
9: -0.1206113, 0.2302211, -0.1130845, 0.2219108, -0.3425221, 0.3433056

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 127
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_B2_A1

### Relational analysis result of IS_A1_A1_B2_B2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3528984, upper bound: 0.3325040
time: 0.88 seconds

## Relational analysis of IS_A1_A1_B2_B2_B2_A2

### Relational analysis result of IS_A1_A1_B2_B2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3258413, upper bound: 0.3234105
time: 0.86 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1755940, 0.1196325, -0.1791032, 0.1223922, -0.2979862, 0.2987357
1: -0.1657065, 0.1249242, -0.1688330, 0.1295736, -0.2952801, 0.2937571
2: -0.1009169, 0.2176833, -0.1044585, 0.2213965, -0.3223134, 0.3221418
3: -0.0986617, 0.2706010, -0.0965597, 0.2790132, -0.3713313, 0.3607632
4: -0.1409538, 0.1588596, -0.1440112, 0.1632273, -0.3041810, 0.3028709
5: -0.1270662, 0.1999935, -0.1307711, 0.2041727, -0.3312389, 0.3307646
6: -0.1633671, 0.1510580, -0.1672360, 0.1551028, -0.3184699, 0.3182940
7: 0.5785890, 1.0723939, 0.5682518, 1.0709462, -0.4923572, 0.5041420
8: -0.1120738, 0.2345980, -0.1161288, 0.2385267, -0.3506004, 0.3507268
9: -0.1203275, 0.2297447, -0.1236560, 0.2337611, -0.3540887, 0.3534007

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3571597, upper bound: 0.3349383
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3695155, upper bound: 0.3746116
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3695155, upper bound: 0.3746116
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1804025, 0.1236696, -0.1755940, 0.1196325, -0.3000349, 0.2992637
1: -0.1701152, 0.1312663, -0.1657065, 0.1249242, -0.2950394, 0.2969728
2: -0.1062283, 0.2229695, -0.1009169, 0.2176833, -0.3239116, 0.3238864
3: -0.1012890, 0.2811121, -0.0986617, 0.2706010, -0.3656042, 0.3734207
4: -0.1452372, 0.1650544, -0.1409538, 0.1588596, -0.3040968, 0.3060082
5: -0.1321446, 0.2058423, -0.1270662, 0.1999935, -0.3321381, 0.3329085
6: -0.1688091, 0.1567670, -0.1633671, 0.1510580, -0.3198671, 0.3201341
7: 0.5654473, 1.0744244, 0.5785890, 1.0723939, -0.5069466, 0.4958354
8: -0.1176876, 0.2403003, -0.1120738, 0.2345980, -0.3522857, 0.3523740
9: -0.1252088, 0.2354038, -0.1203275, 0.2297447, -0.3549535, 0.3557313

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3309278, upper bound: 0.3563240
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B1_B2_B1_A1

### Relational analysis result of IS_A1_A2_B1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3694593, upper bound: 0.3694593
time: 1.01 seconds

## Relational analysis of IS_A1_A2_B1_B2_B1_A2

### Relational analysis result of IS_A1_A2_B1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3694593, upper bound: 0.3694593
time: 0.95 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1755940, 0.1196325, -0.1788291, 0.1224563, -0.2980503, 0.2984616
1: -0.1657065, 0.1249242, -0.1689402, 0.1295181, -0.2952246, 0.2938644
2: -0.1009169, 0.2176833, -0.1050421, 0.2215055, -0.3224224, 0.3227254
3: -0.0986617, 0.2706010, -0.1028352, 0.2789302, -0.3711737, 0.3671718
4: -0.1409538, 0.1588596, -0.1438657, 0.1629859, -0.3039397, 0.3027253
5: -0.1270662, 0.1999935, -0.1306279, 0.2039206, -0.3309867, 0.3306213
6: -0.1633671, 0.1510580, -0.1670104, 0.1552319, -0.3185990, 0.3180683
7: 0.5785890, 1.0723939, 0.5679736, 1.0776608, -0.4990718, 0.5044203
8: -0.1120738, 0.2345980, -0.1157474, 0.2387219, -0.3507957, 0.3503454
9: -0.1203275, 0.2297447, -0.1241060, 0.2336740, -0.3540015, 0.3538507

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3592439, upper bound: 0.3349474
time: 0.95 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B2_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3712368, upper bound: 0.3746116
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3712368, upper bound: 0.3746116
time: 1.06 seconds

## BFS IS instance: IS_A1_A2_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1755940, 0.1196325, -0.1803547, 0.1239133, -0.2995073, 0.2999872
1: -0.1657065, 0.1249242, -0.1704204, 0.1316042, -0.2973107, 0.2953445
2: -0.1009169, 0.2176833, -0.1070789, 0.2233019, -0.3242188, 0.3247622
3: -0.0986617, 0.2706010, -0.1077114, 0.2816291, -0.3738786, 0.3722030
4: -0.1409538, 0.1588596, -0.1452897, 0.1651578, -0.3061115, 0.3041493
5: -0.1270662, 0.1999935, -0.1322553, 0.2058822, -0.3329483, 0.3322488
6: -0.1633671, 0.1510580, -0.1688845, 0.1571378, -0.3205049, 0.3199425
7: 0.5785890, 1.0723939, 0.5644401, 1.0811839, -0.5025949, 0.5079538
8: -0.1120738, 0.2345980, -0.1175945, 0.2407477, -0.3528215, 0.3521925
9: -0.1203275, 0.2297447, -0.1259222, 0.2355606, -0.3558881, 0.3556669

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=32, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3590429, upper bound: 0.3311609
time: 0.96 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A1_A2_B2_B2_A1_B1

### Relational analysis result of IS_A1_A2_B2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3712151, upper bound: 0.3697016
time: 0.93 seconds

## Relational analysis of IS_A1_A2_B2_B2_A1_B2

### Relational analysis result of IS_A1_A2_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3712151, upper bound: 0.3697016
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1788291, 0.1224563, -0.1741697, 0.1182501, -0.2970793, 0.2966260
1: -0.1689402, 0.1295181, -0.1643215, 0.1231771, -0.2921174, 0.2938396
2: -0.1050421, 0.2215055, -0.0990503, 0.2159769, -0.3210190, 0.3205557
3: -0.1028352, 0.2789302, -0.0939293, 0.2683803, -0.3648330, 0.3662128
4: -0.1438657, 0.1629859, -0.1396105, 0.1573178, -0.3011834, 0.3025964
5: -0.1306279, 0.2039206, -0.1255827, 0.1981847, -0.3288125, 0.3295032
6: -0.1670104, 0.1552319, -0.1618320, 0.1492541, -0.3162645, 0.3170639
7: 0.5679736, 1.0776608, 0.5814726, 1.0688863, -0.5009127, 0.4961882
8: -0.1157474, 0.2387219, -0.1105620, 0.2326764, -0.3484238, 0.3492839
9: -0.1241060, 0.2336740, -0.1187423, 0.2279480, -0.3520540, 0.3524162

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3445214, upper bound: 0.3792760
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3781226
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3781226
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1757925, 0.1198970, -0.1674655, 0.1138410, -0.2896335, 0.2873625
1: -0.1661402, 0.1255416, -0.1575197, 0.1119215, -0.2780617, 0.2830613
2: -0.1016698, 0.2181470, -0.0910432, 0.2086956, -0.3103655, 0.3091902
3: -0.1012437, 0.2722510, -0.0890170, 0.2423350, -0.3379869, 0.3538374
4: -0.1411566, 0.1592279, -0.1337435, 0.1498526, -0.2910091, 0.2929714
5: -0.1274135, 0.2002448, -0.1177301, 0.1908531, -0.3182665, 0.3179749
6: -0.1636422, 0.1516037, -0.1544752, 0.1411408, -0.3047830, 0.3060790
7: 0.5763407, 1.0763991, 0.6142168, 1.0700700, -0.4937293, 0.4621823
8: -0.1122721, 0.2351013, -0.1036088, 0.2256446, -0.3379167, 0.3387101
9: -0.1210411, 0.2300825, -0.1114052, 0.2205204, -0.3415616, 0.3414877

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3781226
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3781226
time: 0.92 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1

### Backsubstitution after applying IS history:
0: -0.1754827, 0.1197965, -0.1791032, 0.1223922, -0.2978749, 0.2988997
1: -0.1659405, 0.1252145, -0.1688330, 0.1295736, -0.2955141, 0.2940475
2: -0.1016621, 0.2179262, -0.1044585, 0.2213965, -0.3230587, 0.3223848
3: -0.1051774, 0.2711130, -0.0965597, 0.2790132, -0.3771473, 0.3612323
4: -0.1409440, 0.1589774, -0.1440112, 0.1632273, -0.3041713, 0.3029887
5: -0.1271106, 0.1999557, -0.1307711, 0.2041727, -0.3312833, 0.3307269
6: -0.1634101, 0.1513354, -0.1672360, 0.1551028, -0.3185129, 0.3185714
7: 0.5776169, 1.0792078, 0.5682518, 1.0709462, -0.4933293, 0.5109559
8: -0.1119504, 0.2349345, -0.1161288, 0.2385267, -0.3504770, 0.3510633
9: -0.1209784, 0.2298051, -0.1236560, 0.2337611, -0.3547395, 0.3534610

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3574245, upper bound: 0.3396637
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_B1_A2_A1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3697596, upper bound: 0.3765575
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3697596, upper bound: 0.3765575
time: 1.00 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A2

### Backsubstitution after applying IS history:
0: -0.1684408, 0.1149896, -0.1760962, 0.1198711, -0.2883119, 0.2910858
1: -0.1586944, 0.1134470, -0.1660738, 0.1256117, -0.2843060, 0.2795208
2: -0.0929544, 0.2101773, -0.1011454, 0.2180896, -0.3110439, 0.3113228
3: -0.1004950, 0.2444159, -0.0949562, 0.2723634, -0.3658531, 0.3340673
4: -0.1347493, 0.1510583, -0.1413345, 0.1593573, -0.2941066, 0.2923928
5: -0.1188599, 0.1921550, -0.1275935, 0.2005221, -0.3193820, 0.3197485
6: -0.1556504, 0.1426955, -0.1638376, 0.1515301, -0.3071805, 0.3065331
7: 0.6113134, 1.0804781, 0.5765510, 1.0696836, -0.4583701, 0.5039271
8: -0.1046044, 0.2273157, -0.1126227, 0.2349626, -0.3395670, 0.3399384
9: -0.1130845, 0.2219108, -0.1206113, 0.2302211, -0.3433056, 0.3425221

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 127
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3325040, upper bound: 0.3528984
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B1_A2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3234105, upper bound: 0.3258413
time: 0.82 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1788291, 0.1224563, -0.1755940, 0.1196325, -0.2984616, 0.2980503
1: -0.1689402, 0.1295181, -0.1657065, 0.1249242, -0.2938644, 0.2952246
2: -0.1050421, 0.2215055, -0.1009169, 0.2176833, -0.3227254, 0.3224224
3: -0.1028352, 0.2789302, -0.0986617, 0.2706010, -0.3671717, 0.3711736
4: -0.1438657, 0.1629859, -0.1409538, 0.1588596, -0.3027253, 0.3039397
5: -0.1306279, 0.2039206, -0.1270662, 0.1999935, -0.3306213, 0.3309867
6: -0.1670104, 0.1552319, -0.1633671, 0.1510580, -0.3180683, 0.3185990
7: 0.5679736, 1.0776608, 0.5785890, 1.0723939, -0.5044203, 0.4990718
8: -0.1157474, 0.2387219, -0.1120738, 0.2345980, -0.3503454, 0.3507957
9: -0.1241060, 0.2336740, -0.1203275, 0.2297447, -0.3538507, 0.3540015

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3349474, upper bound: 0.3592439
time: 0.75 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B2_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_B2_A1_B1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3712368
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3712368
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1803547, 0.1239133, -0.1755940, 0.1196325, -0.2999872, 0.2995073
1: -0.1704204, 0.1316042, -0.1657065, 0.1249242, -0.2953445, 0.2973107
2: -0.1070789, 0.2233019, -0.1009169, 0.2176833, -0.3247622, 0.3242188
3: -0.1077114, 0.2816291, -0.0986617, 0.2706010, -0.3722029, 0.3738785
4: -0.1452897, 0.1651578, -0.1409538, 0.1588596, -0.3041493, 0.3061115
5: -0.1322553, 0.2058822, -0.1270662, 0.1999935, -0.3322488, 0.3329483
6: -0.1688845, 0.1571378, -0.1633671, 0.1510580, -0.3199425, 0.3205049
7: 0.5644401, 1.0811839, 0.5785890, 1.0723939, -0.5079538, 0.5025949
8: -0.1175945, 0.2407477, -0.1120738, 0.2345980, -0.3521925, 0.3528215
9: -0.1259222, 0.2355606, -0.1203275, 0.2297447, -0.3556669, 0.3558881

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3311609, upper bound: 0.3590429
time: 0.89 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B1_B2_A2_B1_A1

### Relational analysis result of IS_A2_B1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3697016, upper bound: 0.3712151
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B2_A2_B1_A2

### Relational analysis result of IS_A2_B1_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3697016, upper bound: 0.3712151
time: 0.87 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1739336, 0.1183329, -0.1788291, 0.1224563, -0.2963899, 0.2971621
1: -0.1644406, 0.1231381, -0.1689402, 0.1295181, -0.2939587, 0.2920783
2: -0.0996165, 0.2161084, -0.1050421, 0.2215055, -0.3211220, 0.3211505
3: -0.1003036, 0.2683390, -0.1028352, 0.2789302, -0.3726007, 0.3646174
4: -0.1394941, 0.1572350, -0.1438657, 0.1629859, -0.3024800, 0.3011007
5: -0.1254569, 0.1979885, -0.1306279, 0.2039206, -0.3293775, 0.3286164
6: -0.1616902, 0.1494030, -0.1670104, 0.1552319, -0.3169221, 0.3164134
7: 0.5811852, 1.0756619, 0.5679736, 1.0776608, -0.4964756, 0.5076883
8: -0.1102747, 0.2328919, -0.1157474, 0.2387219, -0.3489966, 0.3486393
9: -0.1192082, 0.2278926, -0.1241060, 0.2336740, -0.3528821, 0.3519987

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.68 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770856, upper bound: 0.3479535
time: 1.01 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3772450, upper bound: 0.3784286
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3772450, upper bound: 0.3784286
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1672865, 0.1138846, -0.1757925, 0.1198970, -0.2871835, 0.2896771
1: -0.1575709, 0.1119591, -0.1661402, 0.1255416, -0.2831125, 0.2780993
2: -0.0914344, 0.2087702, -0.1016698, 0.2181470, -0.3095814, 0.3104401
3: -0.0957292, 0.2423134, -0.1012437, 0.2722510, -0.3605083, 0.3380270
4: -0.1336529, 0.1498031, -0.1411566, 0.1592279, -0.2928809, 0.2909597
5: -0.1176307, 0.1907236, -0.1274135, 0.2002448, -0.3178755, 0.3181371
6: -0.1543910, 0.1412252, -0.1636422, 0.1516037, -0.3059947, 0.3048674
7: 0.6140587, 1.0770576, 0.5763407, 1.0763991, -0.4623404, 0.5007169
8: -0.1033931, 0.2257753, -0.1122721, 0.2351013, -0.3384944, 0.3380474
9: -0.1117706, 0.2204498, -0.1210411, 0.2300825, -0.3418531, 0.3414910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3772450, upper bound: 0.3784286
time: 0.90 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3772450, upper bound: 0.3784286
time: 1.02 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1

### Backsubstitution after applying IS history:
0: -0.1788291, 0.1224563, -0.1754827, 0.1197965, -0.2986256, 0.2979390
1: -0.1689402, 0.1295181, -0.1659405, 0.1252145, -0.2941547, 0.2954586
2: -0.1050421, 0.2215055, -0.1016621, 0.2179262, -0.3229683, 0.3231676
3: -0.1028352, 0.2789302, -0.1051774, 0.2711130, -0.3675253, 0.3768540
4: -0.1438657, 0.1629859, -0.1409440, 0.1589774, -0.3028431, 0.3039299
5: -0.1306279, 0.2039206, -0.1271106, 0.1999557, -0.3305836, 0.3310311
6: -0.1670104, 0.1552319, -0.1634101, 0.1513354, -0.3183458, 0.3186420
7: 0.5679736, 1.0776608, 0.5776169, 1.0792078, -0.5112342, 0.5000439
8: -0.1157474, 0.2387219, -0.1119504, 0.2349345, -0.3506819, 0.3506723
9: -0.1241060, 0.2336740, -0.1209784, 0.2298051, -0.3539111, 0.3546524

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=31, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3390226, upper bound: 0.3609527
time: 0.88 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A1_B2_B1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
time: 1.11 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
time: 0.96 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2

### Backsubstitution after applying IS history:
0: -0.1757925, 0.1198970, -0.1684408, 0.1149896, -0.2907821, 0.2883378
1: -0.1661402, 0.1255416, -0.1586944, 0.1134470, -0.2795872, 0.2842360
2: -0.1016698, 0.2181470, -0.0929544, 0.2101773, -0.3118472, 0.3111013
3: -0.1012437, 0.2722510, -0.1004950, 0.2444159, -0.3403655, 0.3655340
4: -0.1411566, 0.1592279, -0.1347493, 0.1510583, -0.2922148, 0.2939772
5: -0.1274135, 0.2002448, -0.1188599, 0.1921550, -0.3195685, 0.3191046
6: -0.1636422, 0.1516037, -0.1556504, 0.1426955, -0.3063377, 0.3072541
7: 0.5763407, 1.0763991, 0.6113134, 1.0804781, -0.5041373, 0.4650857
8: -0.1122721, 0.2351013, -0.1046044, 0.2273157, -0.3395878, 0.3397057
9: -0.1210411, 0.2300825, -0.1130845, 0.2219108, -0.3429519, 0.3431670

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A1_B2_B2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
time: 0.92 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1754827, 0.1197965, -0.1788291, 0.1224563, -0.2979390, 0.2986256
1: -0.1659405, 0.1252145, -0.1689402, 0.1295181, -0.2954586, 0.2941547
2: -0.1016621, 0.2179262, -0.1050421, 0.2215055, -0.3231676, 0.3229683
3: -0.1051774, 0.2711130, -0.1028352, 0.2789302, -0.3768540, 0.3675253
4: -0.1409440, 0.1589774, -0.1438657, 0.1629859, -0.3039299, 0.3028431
5: -0.1271106, 0.1999557, -0.1306279, 0.2039206, -0.3310311, 0.3305836
6: -0.1634101, 0.1513354, -0.1670104, 0.1552319, -0.3186420, 0.3183458
7: 0.5776169, 1.0792078, 0.5679736, 1.0776608, -0.5000439, 0.5112342
8: -0.1119504, 0.2349345, -0.1157474, 0.2387219, -0.3506723, 0.3506819
9: -0.1209784, 0.2298051, -0.1241060, 0.2336740, -0.3546524, 0.3539111

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=13, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=31, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.41 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 127
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3602216, upper bound: 0.3404973
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3714213, upper bound: 0.3769398
time: 0.99 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3714213, upper bound: 0.3769398
time: 1.05 seconds

## BFS IS instance: IS_A2_B2_A2_B2_B1

### Backsubstitution after applying IS history:
0: -0.1803547, 0.1239133, -0.1754827, 0.1197965, -0.3001512, 0.2993960
1: -0.1704204, 0.1316042, -0.1659405, 0.1252145, -0.2956349, 0.2975447
2: -0.1070789, 0.2233019, -0.1016621, 0.2179262, -0.3250051, 0.3249640
3: -0.1077114, 0.2816291, -0.1051774, 0.2711130, -0.3725501, 0.3795218
4: -0.1452897, 0.1651578, -0.1409440, 0.1589774, -0.3042672, 0.3061018
5: -0.1322553, 0.2058822, -0.1271106, 0.1999557, -0.3322110, 0.3329927
6: -0.1688845, 0.1571378, -0.1634101, 0.1513354, -0.3202199, 0.3205479
7: 0.5644401, 1.0811839, 0.5776169, 1.0792078, -0.5147676, 0.5035670
8: -0.1175945, 0.2407477, -0.1119504, 0.2349345, -0.3525290, 0.3526981
9: -0.1259222, 0.2355606, -0.1209784, 0.2298051, -0.3557273, 0.3565390

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=13, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 127
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3364557, upper bound: 0.3606923
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 127

## Relational analysis of IS_A2_B2_A2_B2_B1_A1

### Relational analysis result of IS_A2_B2_A2_B2_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3713892, upper bound: 0.3723255
time: 1.10 seconds

## Relational analysis of IS_A2_B2_A2_B2_B1_A2

### Relational analysis result of IS_A2_B2_A2_B2_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3713892, upper bound: 0.3723255
time: 1.00 seconds

## Summary of splitting at layer (split count: 5)
- Time for IS candidates: 8.74 seconds
IS_A1_A1_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
IS_A1_A1_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
IS_A1_A1_B1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
IS_A1_A1_B1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3763316
IS_A1_A1_B1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3781226, upper bound: 0.3763316
IS_A1_A1_B1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3781226, upper bound: 0.3763316
IS_A1_A1_B1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3781226, upper bound: 0.3763316
IS_A1_A1_B1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3781226, upper bound: 0.3763316
IS_A1_A1_B2_B1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3695155
IS_A1_A1_B2_B1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3695155
IS_A1_A1_B2_B1_B2_A1, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3456531, upper bound: 0.3263008
IS_A1_A1_B2_B1_B2_A2, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3211644, upper bound: 0.3182779
IS_A1_A1_B2_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3765575, upper bound: 0.3697596
IS_A1_A1_B2_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3765575, upper bound: 0.3697596
IS_A1_A1_B2_B2_B2_A1, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3528984, upper bound: 0.3325040
IS_A1_A1_B2_B2_B2_A2, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3258413, upper bound: 0.3234105
IS_A1_A2_B1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3695155, upper bound: 0.3746116
IS_A1_A2_B1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3695155, upper bound: 0.3746116
IS_A1_A2_B1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3694593, upper bound: 0.3694593
IS_A1_A2_B1_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3694593, upper bound: 0.3694593
IS_A1_A2_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3712368, upper bound: 0.3746116
IS_A1_A2_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3712368, upper bound: 0.3746116
IS_A1_A2_B2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3712151, upper bound: 0.3697016
IS_A1_A2_B2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3712151, upper bound: 0.3697016
IS_A2_B1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3781226
IS_A2_B1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3781226
IS_A2_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3781226
IS_A2_B1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3763316, upper bound: 0.3781226
IS_A2_B1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3697596, upper bound: 0.3765575
IS_A2_B1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3697596, upper bound: 0.3765575
IS_A2_B1_B1_A2_A2_B1, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3325040, upper bound: 0.3528984
IS_A2_B1_B1_A2_A2_B2, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3234105, upper bound: 0.3258413
IS_A2_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3712368
IS_A2_B1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3746116, upper bound: 0.3712368
IS_A2_B1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3697016, upper bound: 0.3712151
IS_A2_B1_B2_A2_B1_A2, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3697016, upper bound: 0.3712151
IS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3772450, upper bound: 0.3784286
IS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3772450, upper bound: 0.3784286
IS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3772450, upper bound: 0.3784286
IS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3772450, upper bound: 0.3784286
IS_A2_B2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
IS_A2_B2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
IS_A2_B2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
IS_A2_B2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3756436, upper bound: 0.3723409
IS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3714213, upper bound: 0.3769398
IS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3714213, upper bound: 0.3769398
IS_A2_B2_A2_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3713892, upper bound: 0.3723255
IS_A2_B2_A2_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 8.74
Output dim: 7, lower bound: -0.3713892, upper bound: 0.3723255

## BFS IS instance: IS_A1_A1_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1741697, 0.1182501, -0.1741697, 0.1182501, -0.2924198, 0.2924198
1: -0.1643215, 0.1231771, -0.1643215, 0.1231771, -0.2874986, 0.2874986
2: -0.0990503, 0.2159769, -0.0990503, 0.2159769, -0.3150272, 0.3150272
3: -0.0939293, 0.2683803, -0.0939293, 0.2683803, -0.3557687, 0.3557687
4: -0.1396105, 0.1573178, -0.1396105, 0.1573178, -0.2969282, 0.2969282
5: -0.1255827, 0.1981847, -0.1255827, 0.1981847, -0.3237674, 0.3237674
6: -0.1618320, 0.1492541, -0.1618320, 0.1492541, -0.3110862, 0.3110862
7: 0.5814726, 1.0688863, 0.5814726, 1.0688863, -0.4874137, 0.4874137
8: -0.1105620, 0.2326764, -0.1105620, 0.2326764, -0.3432384, 0.3432384
9: -0.1187423, 0.2279480, -0.1187423, 0.2279480, -0.3466903, 0.3466903

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3767146, upper bound: 0.3445214
time: 0.90 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3738104, upper bound: 0.3556584
time: 0.94 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3586654, upper bound: 0.3275535
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1741697, 0.1182501, -0.1674655, 0.1138410, -0.2880107, 0.2857156
1: -0.1643215, 0.1231771, -0.1575197, 0.1119215, -0.2762429, 0.2806968
2: -0.0990503, 0.2159769, -0.0910432, 0.2086956, -0.3077459, 0.3070201
3: -0.0939293, 0.2683803, -0.0890170, 0.2423350, -0.3295354, 0.3500551
4: -0.1396105, 0.1573178, -0.1337435, 0.1498526, -0.2894630, 0.2910613
5: -0.1255827, 0.1981847, -0.1177301, 0.1908531, -0.3164358, 0.3159148
6: -0.1618320, 0.1492541, -0.1544752, 0.1411408, -0.3029728, 0.3037294
7: 0.5814726, 1.0688863, 0.6142168, 1.0700700, -0.4885975, 0.4546695
8: -0.1105620, 0.2326764, -0.1036088, 0.2256446, -0.3362066, 0.3362852
9: -0.1187423, 0.2279480, -0.1114052, 0.2205204, -0.3392627, 0.3393533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3741780, upper bound: 0.3338321
time: 0.93 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3586654, upper bound: 0.3275535
time: 1.03 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1674655, 0.1138410, -0.1741697, 0.1182501, -0.2857156, 0.2880107
1: -0.1575197, 0.1119215, -0.1643215, 0.1231771, -0.2806968, 0.2762429
2: -0.0910432, 0.2086956, -0.0990503, 0.2159769, -0.3070201, 0.3077459
3: -0.0890170, 0.2423350, -0.0939293, 0.2683803, -0.3500552, 0.3295353
4: -0.1337435, 0.1498526, -0.1396105, 0.1573178, -0.2910613, 0.2894630
5: -0.1177301, 0.1908531, -0.1255827, 0.1981847, -0.3159148, 0.3164358
6: -0.1544752, 0.1411408, -0.1618320, 0.1492541, -0.3037294, 0.3029728
7: 0.6142168, 1.0700700, 0.5814726, 1.0688863, -0.4546695, 0.4885975
8: -0.1036088, 0.2256446, -0.1105620, 0.2326764, -0.3362852, 0.3362066
9: -0.1114052, 0.2205204, -0.1187423, 0.2279480, -0.3393533, 0.3392627

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.66 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.20 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3313890, upper bound: 0.3480508
time: 0.89 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3234903, upper bound: 0.3234903
time: 0.79 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1674655, 0.1138410, -0.1674655, 0.1138410, -0.2813064, 0.2813064
1: -0.1575197, 0.1119215, -0.1575197, 0.1119215, -0.2694412, 0.2694412
2: -0.0910432, 0.2086956, -0.0910432, 0.2086956, -0.2997388, 0.2997388
3: -0.0890170, 0.2423350, -0.0890170, 0.2423350, -0.3237328, 0.3237328
4: -0.1337435, 0.1498526, -0.1337435, 0.1498526, -0.2835961, 0.2835961
5: -0.1177301, 0.1908531, -0.1177301, 0.1908531, -0.3085831, 0.3085831
6: -0.1544752, 0.1411408, -0.1544752, 0.1411408, -0.2956160, 0.2956160
7: 0.6142168, 1.0700700, 0.6142168, 1.0700700, -0.4558532, 0.4558532
8: -0.1036088, 0.2256446, -0.1036088, 0.2256446, -0.3292534, 0.3292534
9: -0.1114052, 0.2205204, -0.1114052, 0.2205204, -0.3319257, 0.3319257

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3480508, upper bound: 0.3313890
time: 0.92 seconds

## Relational analysis of IS_A1_A1_B1_B1_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3234903, upper bound: 0.3234903
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1741697, 0.1182501, -0.1739336, 0.1183329, -0.2925026, 0.2921838
1: -0.1643215, 0.1231771, -0.1644406, 0.1231381, -0.2874596, 0.2876177
2: -0.0990503, 0.2159769, -0.0996165, 0.2161084, -0.3151587, 0.3155934
3: -0.0939293, 0.2683803, -0.1003036, 0.2683390, -0.3556631, 0.3622701
4: -0.1396105, 0.1573178, -0.1394941, 0.1572350, -0.2968455, 0.2968119
5: -0.1255827, 0.1981847, -0.1254569, 0.1979885, -0.3235712, 0.3236417
6: -0.1618320, 0.1492541, -0.1616902, 0.1494030, -0.3112351, 0.3109444
7: 0.5814726, 1.0688863, 0.5811852, 1.0756619, -0.4941893, 0.4877011
8: -0.1105620, 0.2326764, -0.1102747, 0.2328919, -0.3434539, 0.3429511
9: -0.1187423, 0.2279480, -0.1192082, 0.2278926, -0.3466349, 0.3471562

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3792760, upper bound: 0.3445214
time: 1.04 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3805157, upper bound: 0.3376020
time: 0.98 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3651048, upper bound: 0.3310715
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1741697, 0.1182501, -0.1672865, 0.1138846, -0.2880542, 0.2855366
1: -0.1643215, 0.1231771, -0.1575709, 0.1119591, -0.2762805, 0.2807480
2: -0.0990503, 0.2159769, -0.0914344, 0.2087702, -0.3078205, 0.3074113
3: -0.0939293, 0.2683803, -0.0957292, 0.2423134, -0.3297004, 0.3568669
4: -0.1396105, 0.1573178, -0.1336529, 0.1498031, -0.2894136, 0.2909707
5: -0.1255827, 0.1981847, -0.1176307, 0.1907236, -0.3163063, 0.3158154
6: -0.1618320, 0.1492541, -0.1543910, 0.1412252, -0.3030572, 0.3036452
7: 0.5814726, 1.0688863, 0.6140587, 1.0770576, -0.4955850, 0.4548275
8: -0.1105620, 0.2326764, -0.1033931, 0.2257753, -0.3363373, 0.3360696
9: -0.1187423, 0.2279480, -0.1117706, 0.2204498, -0.3391921, 0.3397186

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3805157, upper bound: 0.3376020
time: 0.98 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3651048, upper bound: 0.3310715
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B1

### Backsubstitution after applying IS history:
0: -0.1674655, 0.1138410, -0.1739336, 0.1183329, -0.2857984, 0.2877746
1: -0.1575197, 0.1119215, -0.1644406, 0.1231381, -0.2806578, 0.2763620
2: -0.0910432, 0.2086956, -0.0996165, 0.2161084, -0.3071516, 0.3083121
3: -0.0890170, 0.2423350, -0.1003036, 0.2683390, -0.3499455, 0.3360367
4: -0.1337435, 0.1498526, -0.1394941, 0.1572350, -0.2909785, 0.2893467
5: -0.1177301, 0.1908531, -0.1254569, 0.1979885, -0.3157186, 0.3163100
6: -0.1544752, 0.1411408, -0.1616902, 0.1494030, -0.3038783, 0.3028310
7: 0.6142168, 1.0700700, 0.5811852, 1.0756619, -0.4614451, 0.4888848
8: -0.1036088, 0.2256446, -0.1102747, 0.2328919, -0.3365008, 0.3359192
9: -0.1114052, 0.2205204, -0.1192082, 0.2278926, -0.3392979, 0.3397286

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3328889, upper bound: 0.3480508
time: 0.83 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3267523, upper bound: 0.3265984
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A2_B2

### Backsubstitution after applying IS history:
0: -0.1674655, 0.1138410, -0.1672865, 0.1138846, -0.2813501, 0.2811275
1: -0.1575197, 0.1119215, -0.1575709, 0.1119591, -0.2694787, 0.2694924
2: -0.0910432, 0.2086956, -0.0914344, 0.2087702, -0.2998134, 0.3001301
3: -0.0890170, 0.2423350, -0.0957292, 0.2423134, -0.3238934, 0.3305184
4: -0.1337435, 0.1498526, -0.1336529, 0.1498031, -0.2835466, 0.2835055
5: -0.1177301, 0.1908531, -0.1176307, 0.1907236, -0.3084537, 0.3084838
6: -0.1544752, 0.1411408, -0.1543910, 0.1412252, -0.2957004, 0.2955318
7: 0.6142168, 1.0700700, 0.6140587, 1.0770576, -0.4628408, 0.4560113
8: -0.1036088, 0.2256446, -0.1033931, 0.2257753, -0.3293841, 0.3290377
9: -0.1114052, 0.2205204, -0.1117706, 0.2204498, -0.3318551, 0.3322910

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_A1

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3543412, upper bound: 0.3350407
time: 1.03 seconds

## Relational analysis of IS_A1_A1_B1_B2_A2_B2_A2

### Relational analysis result of IS_A1_A1_B1_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3267523, upper bound: 0.3265984
time: 0.81 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1741697, 0.1182501, -0.1755940, 0.1196325, -0.2938022, 0.2938442
1: -0.1643215, 0.1231771, -0.1657065, 0.1249242, -0.2892457, 0.2888837
2: -0.0990503, 0.2159769, -0.1009169, 0.2176833, -0.3167336, 0.3168938
3: -0.0939293, 0.2683803, -0.0986617, 0.2706010, -0.3581074, 0.3607296
4: -0.1396105, 0.1573178, -0.1409538, 0.1588596, -0.2984701, 0.2982715
5: -0.1255827, 0.1981847, -0.1270662, 0.1999935, -0.3255761, 0.3252509
6: -0.1618320, 0.1492541, -0.1633671, 0.1510580, -0.3128900, 0.3126213
7: 0.5814726, 1.0688863, 0.5785890, 1.0723939, -0.4909213, 0.4902973
8: -0.1105620, 0.2326764, -0.1120738, 0.2345980, -0.3451600, 0.3447502
9: -0.1187423, 0.2279480, -0.1203275, 0.2297447, -0.3484870, 0.3482755

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3349383, upper bound: 0.3571597
time: 0.86 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3543963, upper bound: 0.3710913
time: 0.87 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3267176, upper bound: 0.3554429
time: 0.91 seconds

## BFS IS instance: IS_A1_A1_B2_B1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1674655, 0.1138410, -0.1755940, 0.1196325, -0.2870980, 0.2894350
1: -0.1575197, 0.1119215, -0.1657065, 0.1249242, -0.2824439, 0.2776280
2: -0.0910432, 0.2086956, -0.1009169, 0.2176833, -0.3087265, 0.3096125
3: -0.0890170, 0.2423350, -0.0986617, 0.2706010, -0.3523687, 0.3344963
4: -0.1337435, 0.1498526, -0.1409538, 0.1588596, -0.2926031, 0.2908064
5: -0.1177301, 0.1908531, -0.1270662, 0.1999935, -0.3177236, 0.3179192
6: -0.1544752, 0.1411408, -0.1633671, 0.1510580, -0.3055332, 0.3045079
7: 0.6142168, 1.0700700, 0.5785890, 1.0723939, -0.4581771, 0.4914810
8: -0.1036088, 0.2256446, -0.1120738, 0.2345980, -0.3382068, 0.3377183
9: -0.1114052, 0.2205204, -0.1203275, 0.2297447, -0.3411500, 0.3408480

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3328174, upper bound: 0.3685669
time: 1.01 seconds

## Relational analysis of IS_A1_A1_B2_B1_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3267176, upper bound: 0.3554429
time: 0.87 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1741697, 0.1182501, -0.1754827, 0.1197965, -0.2939662, 0.2937328
1: -0.1643215, 0.1231771, -0.1659405, 0.1252145, -0.2895360, 0.2891176
2: -0.0990503, 0.2159769, -0.1016621, 0.2179262, -0.3169765, 0.3176391
3: -0.0939293, 0.2683803, -0.1051774, 0.2711130, -0.3585766, 0.3665358
4: -0.1396105, 0.1573178, -0.1409440, 0.1589774, -0.2985879, 0.2982618
5: -0.1255827, 0.1981847, -0.1271106, 0.1999557, -0.3255384, 0.3252953
6: -0.1618320, 0.1492541, -0.1634101, 0.1513354, -0.3131674, 0.3126642
7: 0.5814726, 1.0688863, 0.5776169, 1.0792078, -0.4977352, 0.4912694
8: -0.1105620, 0.2326764, -0.1119504, 0.2349345, -0.3454965, 0.3446268
9: -0.1187423, 0.2279480, -0.1209784, 0.2298051, -0.3485473, 0.3489264

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3396637, upper bound: 0.3574245
time: 0.91 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3585447, upper bound: 0.3714677
time: 0.92 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A1_A2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3299857, upper bound: 0.3560712
time: 0.92 seconds

## BFS IS instance: IS_A1_A1_B2_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1674655, 0.1138410, -0.1754827, 0.1197965, -0.2872620, 0.2893236
1: -0.1575197, 0.1119215, -0.1659405, 0.1252145, -0.2827342, 0.2778619
2: -0.0910432, 0.2086956, -0.1016621, 0.2179262, -0.3089694, 0.3103577
3: -0.0890170, 0.2423350, -0.1051774, 0.2711130, -0.3528658, 0.3402941
4: -0.1337435, 0.1498526, -0.1409440, 0.1589774, -0.2927209, 0.2907966
5: -0.1177301, 0.1908531, -0.1271106, 0.1999557, -0.3176858, 0.3179636
6: -0.1544752, 0.1411408, -0.1634101, 0.1513354, -0.3058106, 0.3045509
7: 0.6142168, 1.0700700, 0.5776169, 1.0792078, -0.4649910, 0.4924531
8: -0.1036088, 0.2256446, -0.1119504, 0.2349345, -0.3385434, 0.3375949
9: -0.1114052, 0.2205204, -0.1209784, 0.2298051, -0.3412103, 0.3414989

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_B1

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3352002, upper bound: 0.3685669
time: 0.97 seconds

## Relational analysis of IS_A1_A1_B2_B2_B1_A2_B2

### Relational analysis result of IS_A1_A1_B2_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3299857, upper bound: 0.3560712
time: 0.91 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1755940, 0.1196325, -0.1741697, 0.1182501, -0.2938442, 0.2938022
1: -0.1657065, 0.1249242, -0.1643215, 0.1231771, -0.2888837, 0.2892457
2: -0.1009169, 0.2176833, -0.0990503, 0.2159769, -0.3168938, 0.3167336
3: -0.0986617, 0.2706010, -0.0939293, 0.2683803, -0.3607297, 0.3581074
4: -0.1409538, 0.1588596, -0.1396105, 0.1573178, -0.2982715, 0.2984701
5: -0.1270662, 0.1999935, -0.1255827, 0.1981847, -0.3252509, 0.3255761
6: -0.1633671, 0.1510580, -0.1618320, 0.1492541, -0.3126213, 0.3128900
7: 0.5785890, 1.0723939, 0.5814726, 1.0688863, -0.4902973, 0.4909213
8: -0.1120738, 0.2345980, -0.1105620, 0.2326764, -0.3447502, 0.3451600
9: -0.1203275, 0.2297447, -0.1187423, 0.2279480, -0.3482755, 0.3484870

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3571597, upper bound: 0.3349383
time: 0.89 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3710913, upper bound: 0.3543963
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B1_B2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3554429, upper bound: 0.3267176
time: 0.93 seconds

## BFS IS instance: IS_A1_A2_B1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1755940, 0.1196325, -0.1674655, 0.1138410, -0.2894350, 0.2870980
1: -0.1657065, 0.1249242, -0.1575197, 0.1119215, -0.2776280, 0.2824439
2: -0.1009169, 0.2176833, -0.0910432, 0.2086956, -0.3096125, 0.3087265
3: -0.0986617, 0.2706010, -0.0890170, 0.2423350, -0.3344963, 0.3523686
4: -0.1409538, 0.1588596, -0.1337435, 0.1498526, -0.2908064, 0.2926031
5: -0.1270662, 0.1999935, -0.1177301, 0.1908531, -0.3179192, 0.3177236
6: -0.1633671, 0.1510580, -0.1544752, 0.1411408, -0.3045079, 0.3055332
7: 0.5785890, 1.0723939, 0.6142168, 1.0700700, -0.4914810, 0.4581771
8: -0.1120738, 0.2345980, -0.1036088, 0.2256446, -0.3377183, 0.3382068
9: -0.1203275, 0.2297447, -0.1114052, 0.2205204, -0.3408480, 0.3411500

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3685669, upper bound: 0.3328174
time: 1.05 seconds

## Relational analysis of IS_A1_A2_B1_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3554429, upper bound: 0.3267176
time: 0.88 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1755940, 0.1196325, -0.1739336, 0.1183329, -0.2939270, 0.2935661
1: -0.1657065, 0.1249242, -0.1644406, 0.1231381, -0.2888446, 0.2893648
2: -0.1009169, 0.2176833, -0.0996165, 0.2161084, -0.3170253, 0.3172998
3: -0.0986617, 0.2706010, -0.1003036, 0.2683390, -0.3606240, 0.3646089
4: -0.1409538, 0.1588596, -0.1394941, 0.1572350, -0.2981888, 0.2983538
5: -0.1270662, 0.1999935, -0.1254569, 0.1979885, -0.3250547, 0.3254504
6: -0.1633671, 0.1510580, -0.1616902, 0.1494030, -0.3127702, 0.3127482
7: 0.5785890, 1.0723939, 0.5811852, 1.0756619, -0.4970729, 0.4912087
8: -0.1120738, 0.2345980, -0.1102747, 0.2328919, -0.3449657, 0.3448727
9: -0.1203275, 0.2297447, -0.1192082, 0.2278926, -0.3482202, 0.3489529

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3592439, upper bound: 0.3349474
time: 0.95 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3755019, upper bound: 0.3364038
time: 0.95 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3619098, upper bound: 0.3301412
time: 0.87 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1755940, 0.1196325, -0.1672865, 0.1138846, -0.2894786, 0.2869190
1: -0.1657065, 0.1249242, -0.1575709, 0.1119591, -0.2776656, 0.2824951
2: -0.1009169, 0.2176833, -0.0914344, 0.2087702, -0.3096871, 0.3091177
3: -0.0986617, 0.2706010, -0.0957292, 0.2423134, -0.3346612, 0.3591803
4: -0.1409538, 0.1588596, -0.1336529, 0.1498031, -0.2907569, 0.2925126
5: -0.1270662, 0.1999935, -0.1176307, 0.1907236, -0.3177898, 0.3176242
6: -0.1633671, 0.1510580, -0.1543910, 0.1412252, -0.3045923, 0.3054490
7: 0.5785890, 1.0723939, 0.6140587, 1.0770576, -0.4984686, 0.4583352
8: -0.1120738, 0.2345980, -0.1033931, 0.2257753, -0.3378491, 0.3379912
9: -0.1203275, 0.2297447, -0.1117706, 0.2204498, -0.3407774, 0.3415153

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3755019, upper bound: 0.3364038
time: 0.91 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3619098, upper bound: 0.3301412
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1739336, 0.1183329, -0.1741697, 0.1182501, -0.2921838, 0.2925026
1: -0.1644406, 0.1231381, -0.1643215, 0.1231771, -0.2876177, 0.2874596
2: -0.0996165, 0.2161084, -0.0990503, 0.2159769, -0.3155934, 0.3151587
3: -0.1003036, 0.2683390, -0.0939293, 0.2683803, -0.3622701, 0.3556630
4: -0.1394941, 0.1572350, -0.1396105, 0.1573178, -0.2968119, 0.2968455
5: -0.1254569, 0.1979885, -0.1255827, 0.1981847, -0.3236417, 0.3235712
6: -0.1616902, 0.1494030, -0.1618320, 0.1492541, -0.3109444, 0.3112351
7: 0.5811852, 1.0756619, 0.5814726, 1.0688863, -0.4877011, 0.4941893
8: -0.1102747, 0.2328919, -0.1105620, 0.2326764, -0.3429511, 0.3434539
9: -0.1192082, 0.2278926, -0.1187423, 0.2279480, -0.3471562, 0.3466349

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.47 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3445214, upper bound: 0.3792760
time: 0.85 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3376020, upper bound: 0.3805157
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3310715, upper bound: 0.3651048
time: 0.76 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1672865, 0.1138846, -0.1741697, 0.1182501, -0.2855366, 0.2880542
1: -0.1575709, 0.1119591, -0.1643215, 0.1231771, -0.2807480, 0.2762805
2: -0.0914344, 0.2087702, -0.0990503, 0.2159769, -0.3074113, 0.3078205
3: -0.0957292, 0.2423134, -0.0939293, 0.2683803, -0.3568669, 0.3297004
4: -0.1336529, 0.1498031, -0.1396105, 0.1573178, -0.2909707, 0.2894136
5: -0.1176307, 0.1907236, -0.1255827, 0.1981847, -0.3158154, 0.3163063
6: -0.1543910, 0.1412252, -0.1618320, 0.1492541, -0.3036452, 0.3030572
7: 0.6140587, 1.0770576, 0.5814726, 1.0688863, -0.4548275, 0.4955850
8: -0.1033931, 0.2257753, -0.1105620, 0.2326764, -0.3360696, 0.3363373
9: -0.1117706, 0.2204498, -0.1187423, 0.2279480, -0.3397186, 0.3391921

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3376020, upper bound: 0.3805157
time: 0.99 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3310715, upper bound: 0.3651048
time: 0.89 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1739336, 0.1183329, -0.1674655, 0.1138410, -0.2877746, 0.2857984
1: -0.1644406, 0.1231381, -0.1575197, 0.1119215, -0.2763620, 0.2806578
2: -0.0996165, 0.2161084, -0.0910432, 0.2086956, -0.3083121, 0.3071516
3: -0.1003036, 0.2683390, -0.0890170, 0.2423350, -0.3360367, 0.3499456
4: -0.1394941, 0.1572350, -0.1337435, 0.1498526, -0.2893467, 0.2909785
5: -0.1254569, 0.1979885, -0.1177301, 0.1908531, -0.3163100, 0.3157186
6: -0.1616902, 0.1494030, -0.1544752, 0.1411408, -0.3028310, 0.3038783
7: 0.5811852, 1.0756619, 0.6142168, 1.0700700, -0.4888848, 0.4614451
8: -0.1102747, 0.2328919, -0.1036088, 0.2256446, -0.3359192, 0.3365008
9: -0.1192082, 0.2278926, -0.1114052, 0.2205204, -0.3397286, 0.3392979

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.46 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3480508, upper bound: 0.3328889
time: 0.87 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3265984, upper bound: 0.3267523
time: 0.81 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B2_A2

### Backsubstitution after applying IS history:
0: -0.1672865, 0.1138846, -0.1674655, 0.1138410, -0.2811275, 0.2813501
1: -0.1575709, 0.1119591, -0.1575197, 0.1119215, -0.2694924, 0.2694787
2: -0.0914344, 0.2087702, -0.0910432, 0.2086956, -0.3001301, 0.2998134
3: -0.0957292, 0.2423134, -0.0890170, 0.2423350, -0.3305184, 0.3238935
4: -0.1336529, 0.1498031, -0.1337435, 0.1498526, -0.2835055, 0.2835466
5: -0.1176307, 0.1907236, -0.1177301, 0.1908531, -0.3084838, 0.3084537
6: -0.1543910, 0.1412252, -0.1544752, 0.1411408, -0.2955318, 0.2957004
7: 0.6140587, 1.0770576, 0.6142168, 1.0700700, -0.4560113, 0.4628408
8: -0.1033931, 0.2257753, -0.1036088, 0.2256446, -0.3290377, 0.3293841
9: -0.1117706, 0.2204498, -0.1114052, 0.2205204, -0.3322910, 0.3318551

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B1

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3350407, upper bound: 0.3543412
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B1_A1_B2_A2_B2

### Relational analysis result of IS_A2_B1_B1_A1_B2_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3265984, upper bound: 0.3267523
time: 0.86 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B1

### Backsubstitution after applying IS history:
0: -0.1754827, 0.1197965, -0.1741697, 0.1182501, -0.2937328, 0.2939662
1: -0.1659405, 0.1252145, -0.1643215, 0.1231771, -0.2891176, 0.2895360
2: -0.1016621, 0.2179262, -0.0990503, 0.2159769, -0.3176391, 0.3169765
3: -0.1051774, 0.2711130, -0.0939293, 0.2683803, -0.3665357, 0.3585766
4: -0.1409440, 0.1589774, -0.1396105, 0.1573178, -0.2982618, 0.2985879
5: -0.1271106, 0.1999557, -0.1255827, 0.1981847, -0.3252953, 0.3255384
6: -0.1634101, 0.1513354, -0.1618320, 0.1492541, -0.3126642, 0.3131674
7: 0.5776169, 1.0792078, 0.5814726, 1.0688863, -0.4912694, 0.4977352
8: -0.1119504, 0.2349345, -0.1105620, 0.2326764, -0.3446268, 0.3454965
9: -0.1209784, 0.2298051, -0.1187423, 0.2279480, -0.3489264, 0.3485473

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3574245, upper bound: 0.3396637
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3714677, upper bound: 0.3585447
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B1_B2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3560712, upper bound: 0.3299857
time: 0.91 seconds

## BFS IS instance: IS_A2_B1_B1_A2_A1_B2

### Backsubstitution after applying IS history:
0: -0.1754827, 0.1197965, -0.1674655, 0.1138410, -0.2893236, 0.2872620
1: -0.1659405, 0.1252145, -0.1575197, 0.1119215, -0.2778619, 0.2827342
2: -0.1016621, 0.2179262, -0.0910432, 0.2086956, -0.3103577, 0.3089694
3: -0.1051774, 0.2711130, -0.0890170, 0.2423350, -0.3402941, 0.3528658
4: -0.1409440, 0.1589774, -0.1337435, 0.1498526, -0.2907966, 0.2927209
5: -0.1271106, 0.1999557, -0.1177301, 0.1908531, -0.3179636, 0.3176858
6: -0.1634101, 0.1513354, -0.1544752, 0.1411408, -0.3045509, 0.3058106
7: 0.5776169, 1.0792078, 0.6142168, 1.0700700, -0.4924531, 0.4649910
8: -0.1119504, 0.2349345, -0.1036088, 0.2256446, -0.3375949, 0.3385434
9: -0.1209784, 0.2298051, -0.1114052, 0.2205204, -0.3414989, 0.3412103

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.58 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A1

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3685669, upper bound: 0.3352002
time: 0.95 seconds

## Relational analysis of IS_A2_B1_B1_A2_A1_B2_A2

### Relational analysis result of IS_A2_B1_B1_A2_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3560712, upper bound: 0.3299857
time: 0.95 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1739336, 0.1183329, -0.1755940, 0.1196325, -0.2935661, 0.2939270
1: -0.1644406, 0.1231381, -0.1657065, 0.1249242, -0.2893648, 0.2888446
2: -0.0996165, 0.2161084, -0.1009169, 0.2176833, -0.3172998, 0.3170253
3: -0.1003036, 0.2683390, -0.0986617, 0.2706010, -0.3646088, 0.3606239
4: -0.1394941, 0.1572350, -0.1409538, 0.1588596, -0.2983538, 0.2981888
5: -0.1254569, 0.1979885, -0.1270662, 0.1999935, -0.3254504, 0.3250547
6: -0.1616902, 0.1494030, -0.1633671, 0.1510580, -0.3127482, 0.3127702
7: 0.5811852, 1.0756619, 0.5785890, 1.0723939, -0.4912087, 0.4970729
8: -0.1102747, 0.2328919, -0.1120738, 0.2345980, -0.3448727, 0.3449657
9: -0.1192082, 0.2278926, -0.1203275, 0.2297447, -0.3489529, 0.3482202

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.53 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3349474, upper bound: 0.3592439
time: 0.86 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3364038, upper bound: 0.3755019
time: 0.94 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3301412, upper bound: 0.3619098
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A2

### Backsubstitution after applying IS history:
0: -0.1672865, 0.1138846, -0.1755940, 0.1196325, -0.2869190, 0.2894786
1: -0.1575709, 0.1119591, -0.1657065, 0.1249242, -0.2824951, 0.2776656
2: -0.0914344, 0.2087702, -0.1009169, 0.2176833, -0.3091177, 0.3096871
3: -0.0957292, 0.2423134, -0.0986617, 0.2706010, -0.3591804, 0.3346613
4: -0.1336529, 0.1498031, -0.1409538, 0.1588596, -0.2925126, 0.2907569
5: -0.1176307, 0.1907236, -0.1270662, 0.1999935, -0.3176242, 0.3177898
6: -0.1543910, 0.1412252, -0.1633671, 0.1510580, -0.3054490, 0.3045923
7: 0.6140587, 1.0770576, 0.5785890, 1.0723939, -0.4583352, 0.4984686
8: -0.1033931, 0.2257753, -0.1120738, 0.2345980, -0.3379912, 0.3378491
9: -0.1117706, 0.2204498, -0.1203275, 0.2297447, -0.3415153, 0.3407774

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B1

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3364038, upper bound: 0.3755019
time: 0.91 seconds

## Relational analysis of IS_A2_B1_B2_A1_B1_A2_B2

### Relational analysis result of IS_A2_B1_B2_A1_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3301412, upper bound: 0.3619098
time: 0.97 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1739336, 0.1183329, -0.1739336, 0.1183329, -0.2922665, 0.2922665
1: -0.1644406, 0.1231381, -0.1644406, 0.1231381, -0.2875786, 0.2875786
2: -0.0996165, 0.2161084, -0.0996165, 0.2161084, -0.3157249, 0.3157249
3: -0.1003036, 0.2683390, -0.1003036, 0.2683390, -0.3620541, 0.3620541
4: -0.1394941, 0.1572350, -0.1394941, 0.1572350, -0.2967291, 0.2967291
5: -0.1254569, 0.1979885, -0.1254569, 0.1979885, -0.3234454, 0.3234454
6: -0.1616902, 0.1494030, -0.1616902, 0.1494030, -0.3110933, 0.3110933
7: 0.5811852, 1.0756619, 0.5811852, 1.0756619, -0.4944767, 0.4944767
8: -0.1102747, 0.2328919, -0.1102747, 0.2328919, -0.3431666, 0.3431666
9: -0.1192082, 0.2278926, -0.1192082, 0.2278926, -0.3471008, 0.3471008

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770856, upper bound: 0.3479535
time: 1.07 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3787953, upper bound: 0.3408715
time: 0.95 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3646159, upper bound: 0.3356805
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1739336, 0.1183329, -0.1672865, 0.1138846, -0.2878182, 0.2856194
1: -0.1644406, 0.1231381, -0.1575709, 0.1119591, -0.2763996, 0.2807090
2: -0.0996165, 0.2161084, -0.0914344, 0.2087702, -0.3083867, 0.3075429
3: -0.1003036, 0.2683390, -0.0957292, 0.2423134, -0.3360830, 0.3566185
4: -0.1394941, 0.1572350, -0.1336529, 0.1498031, -0.2892973, 0.2908880
5: -0.1254569, 0.1979885, -0.1176307, 0.1907236, -0.3161806, 0.3156192
6: -0.1616902, 0.1494030, -0.1543910, 0.1412252, -0.3029154, 0.3037941
7: 0.5811852, 1.0756619, 0.6140587, 1.0770576, -0.4958724, 0.4616032
8: -0.1102747, 0.2328919, -0.1033931, 0.2257753, -0.3360500, 0.3362851
9: -0.1192082, 0.2278926, -0.1117706, 0.2204498, -0.3396580, 0.3396632

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3787953, upper bound: 0.3408715
time: 0.98 seconds

## Relational analysis of IS_A2_B2_A1_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A1_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3646159, upper bound: 0.3356805
time: 0.95 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1672865, 0.1138846, -0.1739336, 0.1183329, -0.2856194, 0.2878182
1: -0.1575709, 0.1119591, -0.1644406, 0.1231381, -0.2807090, 0.2763996
2: -0.0914344, 0.2087702, -0.0996165, 0.2161084, -0.3075429, 0.3083867
3: -0.0957292, 0.2423134, -0.1003036, 0.2683390, -0.3566184, 0.3360829
4: -0.1336529, 0.1498031, -0.1394941, 0.1572350, -0.2908880, 0.2892973
5: -0.1176307, 0.1907236, -0.1254569, 0.1979885, -0.3156192, 0.3161806
6: -0.1543910, 0.1412252, -0.1616902, 0.1494030, -0.3037941, 0.3029154
7: 0.6140587, 1.0770576, 0.5811852, 1.0756619, -0.4616032, 0.4958724
8: -0.1033931, 0.2257753, -0.1102747, 0.2328919, -0.3362851, 0.3360500
9: -0.1117706, 0.2204498, -0.1192082, 0.2278926, -0.3396632, 0.3396580

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3372537, upper bound: 0.3550138
time: 0.96 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3311066, upper bound: 0.3321747
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A1_B1_A2_B2

### Backsubstitution after applying IS history:
0: -0.1672865, 0.1138846, -0.1672865, 0.1138846, -0.2811710, 0.2811710
1: -0.1575709, 0.1119591, -0.1575709, 0.1119591, -0.2695299, 0.2695299
2: -0.0914344, 0.2087702, -0.0914344, 0.2087702, -0.3002046, 0.3002046
3: -0.0957292, 0.2423134, -0.0957292, 0.2423134, -0.3305476, 0.3305476
4: -0.1336529, 0.1498031, -0.1336529, 0.1498031, -0.2834561, 0.2834561
5: -0.1176307, 0.1907236, -0.1176307, 0.1907236, -0.3083544, 0.3083544
6: -0.1543910, 0.1412252, -0.1543910, 0.1412252, -0.2956162, 0.2956162
7: 0.6140587, 1.0770576, 0.6140587, 1.0770576, -0.4629989, 0.4629989
8: -0.1033931, 0.2257753, -0.1033931, 0.2257753, -0.3291685, 0.3291685
9: -0.1117706, 0.2204498, -0.1117706, 0.2204498, -0.3322204, 0.3322204

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.51 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_B1

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3372537, upper bound: 0.3550138
time: 0.86 seconds

## Relational analysis of IS_A2_B2_A1_B1_A2_B2_B2

### Relational analysis result of IS_A2_B2_A1_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3311066, upper bound: 0.3321747
time: 0.84 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_A1

### Backsubstitution after applying IS history:
0: -0.1739336, 0.1183329, -0.1754827, 0.1197965, -0.2937301, 0.2938156
1: -0.1644406, 0.1231381, -0.1659405, 0.1252145, -0.2896551, 0.2890785
2: -0.0996165, 0.2161084, -0.1016621, 0.2179262, -0.3175428, 0.3177705
3: -0.1003036, 0.2683390, -0.1051774, 0.2711130, -0.3649620, 0.3662947
4: -0.1394941, 0.1572350, -0.1409440, 0.1589774, -0.2984716, 0.2981790
5: -0.1254569, 0.1979885, -0.1271106, 0.1999557, -0.3254127, 0.3250991
6: -0.1616902, 0.1494030, -0.1634101, 0.1513354, -0.3130256, 0.3128131
7: 0.5811852, 1.0756619, 0.5776169, 1.0792078, -0.4980226, 0.4980450
8: -0.1102747, 0.2328919, -0.1119504, 0.2349345, -0.3452092, 0.3448423
9: -0.1192082, 0.2278926, -0.1209784, 0.2298051, -0.3490132, 0.3488711

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3390226, upper bound: 0.3609527
time: 0.93 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3391580, upper bound: 0.3762956
time: 1.43 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A1_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3339615, upper bound: 0.3642453
time: 1.10 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B1_A2

### Backsubstitution after applying IS history:
0: -0.1672865, 0.1138846, -0.1754827, 0.1197965, -0.2870830, 0.2893672
1: -0.1575709, 0.1119591, -0.1659405, 0.1252145, -0.2827854, 0.2778995
2: -0.0914344, 0.2087702, -0.1016621, 0.2179262, -0.3093607, 0.3104324
3: -0.0957292, 0.2423134, -0.1051774, 0.2711130, -0.3595325, 0.3403305
4: -0.1336529, 0.1498031, -0.1409440, 0.1589774, -0.2926304, 0.2907472
5: -0.1176307, 0.1907236, -0.1271106, 0.1999557, -0.3175865, 0.3178342
6: -0.1543910, 0.1412252, -0.1634101, 0.1513354, -0.3057264, 0.3046353
7: 0.6140587, 1.0770576, 0.5776169, 1.0792078, -0.4651490, 0.4994407
8: -0.1033931, 0.2257753, -0.1119504, 0.2349345, -0.3383277, 0.3377257
9: -0.1117706, 0.2204498, -0.1209784, 0.2298051, -0.3415757, 0.3414282

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.65 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_B1

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3391580, upper bound: 0.3762956
time: 1.03 seconds

## Relational analysis of IS_A2_B2_A1_B2_B1_A2_B2

### Relational analysis result of IS_A2_B2_A1_B2_B1_A2_B2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3339615, upper bound: 0.3642453
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_A1

### Backsubstitution after applying IS history:
0: -0.1739336, 0.1183329, -0.1684408, 0.1149896, -0.2889232, 0.2867737
1: -0.1644406, 0.1231381, -0.1586944, 0.1134470, -0.2778875, 0.2818325
2: -0.0996165, 0.2161084, -0.0929544, 0.2101773, -0.3097939, 0.3090628
3: -0.1003036, 0.2683390, -0.1004950, 0.2444159, -0.3383710, 0.3616442
4: -0.1394941, 0.1572350, -0.1347493, 0.1510583, -0.2905524, 0.2919843
5: -0.1254569, 0.1979885, -0.1188599, 0.1921550, -0.3176119, 0.3168484
6: -0.1616902, 0.1494030, -0.1556504, 0.1426955, -0.3043857, 0.3050535
7: 0.5811852, 1.0756619, 0.6113134, 1.0804781, -0.4992929, 0.4643485
8: -0.1102747, 0.2328919, -0.1046044, 0.2273157, -0.3375904, 0.3374963
9: -0.1192082, 0.2278926, -0.1130845, 0.2219108, -0.3411190, 0.3409772

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3518887, upper bound: 0.3360834
time: 0.89 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A1_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3305552, upper bound: 0.3301891
time: 0.89 seconds

## BFS IS instance: IS_A2_B2_A1_B2_B2_A2

### Backsubstitution after applying IS history:
0: -0.1672865, 0.1138846, -0.1684408, 0.1149896, -0.2822761, 0.2823253
1: -0.1575709, 0.1119591, -0.1586944, 0.1134470, -0.2710178, 0.2706534
2: -0.0914344, 0.2087702, -0.0929544, 0.2101773, -0.3016118, 0.3017246
3: -0.0957292, 0.2423134, -0.1004950, 0.2444159, -0.3328876, 0.3355821
4: -0.1336529, 0.1498031, -0.1347493, 0.1510583, -0.2847112, 0.2845524
5: -0.1176307, 0.1907236, -0.1188599, 0.1921550, -0.3097857, 0.3095835
6: -0.1543910, 0.1412252, -0.1556504, 0.1426955, -0.2970865, 0.2968756
7: 0.6140587, 1.0770576, 0.6113134, 1.0804781, -0.4664193, 0.4657442
8: -0.1033931, 0.2257753, -0.1046044, 0.2273157, -0.3307089, 0.3303797
9: -0.1117706, 0.2204498, -0.1130845, 0.2219108, -0.3336814, 0.3335344

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.64 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_A1

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3518887, upper bound: 0.3360834
time: 0.92 seconds

## Relational analysis of IS_A2_B2_A1_B2_B2_A2_A2

### Relational analysis result of IS_A2_B2_A1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3305552, upper bound: 0.3301891
time: 0.90 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1754827, 0.1197965, -0.1739336, 0.1183329, -0.2938156, 0.2937301
1: -0.1659405, 0.1252145, -0.1644406, 0.1231381, -0.2890785, 0.2896551
2: -0.1016621, 0.2179262, -0.0996165, 0.2161084, -0.3177705, 0.3175428
3: -0.1051774, 0.2711130, -0.1003036, 0.2683390, -0.3662946, 0.3649620
4: -0.1409440, 0.1589774, -0.1394941, 0.1572350, -0.2981790, 0.2984716
5: -0.1271106, 0.1999557, -0.1254569, 0.1979885, -0.3250991, 0.3254127
6: -0.1634101, 0.1513354, -0.1616902, 0.1494030, -0.3128131, 0.3130256
7: 0.5776169, 1.0792078, 0.5811852, 1.0756619, -0.4980450, 0.4980226
8: -0.1119504, 0.2349345, -0.1102747, 0.2328919, -0.3448423, 0.3452092
9: -0.1209784, 0.2298051, -0.1192082, 0.2278926, -0.3488711, 0.3490132

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.55 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_B1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3602216, upper bound: 0.3404973
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3742318, upper bound: 0.3404907
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B1_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3630696, upper bound: 0.3353802
time: 0.91 seconds

## BFS IS instance: IS_A2_B2_A2_B1_A1_B2

### Backsubstitution after applying IS history:
0: -0.1754827, 0.1197965, -0.1672865, 0.1138846, -0.2893672, 0.2870830
1: -0.1659405, 0.1252145, -0.1575709, 0.1119591, -0.2778995, 0.2827854
2: -0.1016621, 0.2179262, -0.0914344, 0.2087702, -0.3104324, 0.3093607
3: -0.1051774, 0.2711130, -0.0957292, 0.2423134, -0.3403305, 0.3595325
4: -0.1409440, 0.1589774, -0.1336529, 0.1498031, -0.2907472, 0.2926304
5: -0.1271106, 0.1999557, -0.1176307, 0.1907236, -0.3178342, 0.3175865
6: -0.1634101, 0.1513354, -0.1543910, 0.1412252, -0.3046353, 0.3057264
7: 0.5776169, 1.0792078, 0.6140587, 1.0770576, -0.4994407, 0.4651490
8: -0.1119504, 0.2349345, -0.1033931, 0.2257753, -0.3377257, 0.3383277
9: -0.1209784, 0.2298051, -0.1117706, 0.2204498, -0.3414282, 0.3415757

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A1

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3742318, upper bound: 0.3404907
time: 0.94 seconds

## Relational analysis of IS_A2_B2_A2_B1_A1_B2_A2

### Relational analysis result of IS_A2_B2_A2_B1_A1_B2_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3630696, upper bound: 0.3353802
time: 0.94 seconds

## Summary of splitting at layer (split count: 6)
- Time for IS candidates: 5.67 seconds
IS_A1_A1_B1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3738104, upper bound: 0.3556584
IS_A1_A1_B1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3586654, upper bound: 0.3275535
IS_A1_A1_B1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3741780, upper bound: 0.3338321
IS_A1_A1_B1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3586654, upper bound: 0.3275535
IS_A1_A1_B1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3313890, upper bound: 0.3480508
IS_A1_A1_B1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3234903, upper bound: 0.3234903
IS_A1_A1_B1_B1_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3480508, upper bound: 0.3313890
IS_A1_A1_B1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3234903, upper bound: 0.3234903
IS_A1_A1_B1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3805157, upper bound: 0.3376020
IS_A1_A1_B1_B2_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3651048, upper bound: 0.3310715
IS_A1_A1_B1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3805157, upper bound: 0.3376020
IS_A1_A1_B1_B2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3651048, upper bound: 0.3310715
IS_A1_A1_B1_B2_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3328889, upper bound: 0.3480508
IS_A1_A1_B1_B2_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3267523, upper bound: 0.3265984
IS_A1_A1_B1_B2_A2_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3543412, upper bound: 0.3350407
IS_A1_A1_B1_B2_A2_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3267523, upper bound: 0.3265984
IS_A1_A1_B2_B1_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3543963, upper bound: 0.3710913
IS_A1_A1_B2_B1_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3267176, upper bound: 0.3554429
IS_A1_A1_B2_B1_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3328174, upper bound: 0.3685669
IS_A1_A1_B2_B1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3267176, upper bound: 0.3554429
IS_A1_A1_B2_B2_B1_A1_A1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3585447, upper bound: 0.3714677
IS_A1_A1_B2_B2_B1_A1_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3299857, upper bound: 0.3560712
IS_A1_A1_B2_B2_B1_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3352002, upper bound: 0.3685669
IS_A1_A1_B2_B2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3299857, upper bound: 0.3560712
IS_A1_A2_B1_B1_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3710913, upper bound: 0.3543963
IS_A1_A2_B1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3554429, upper bound: 0.3267176
IS_A1_A2_B1_B1_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3685669, upper bound: 0.3328174
IS_A1_A2_B1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3554429, upper bound: 0.3267176
IS_A1_A2_B2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3755019, upper bound: 0.3364038
IS_A1_A2_B2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3619098, upper bound: 0.3301412
IS_A1_A2_B2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3755019, upper bound: 0.3364038
IS_A1_A2_B2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3619098, upper bound: 0.3301412
IS_A2_B1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3376020, upper bound: 0.3805157
IS_A2_B1_B1_A1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3310715, upper bound: 0.3651048
IS_A2_B1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3376020, upper bound: 0.3805157
IS_A2_B1_B1_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3310715, upper bound: 0.3651048
IS_A2_B1_B1_A1_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3480508, upper bound: 0.3328889
IS_A2_B1_B1_A1_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3265984, upper bound: 0.3267523
IS_A2_B1_B1_A1_B2_A2_B1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3350407, upper bound: 0.3543412
IS_A2_B1_B1_A1_B2_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3265984, upper bound: 0.3267523
IS_A2_B1_B1_A2_A1_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3714677, upper bound: 0.3585447
IS_A2_B1_B1_A2_A1_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3560712, upper bound: 0.3299857
IS_A2_B1_B1_A2_A1_B2_A1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3685669, upper bound: 0.3352002
IS_A2_B1_B1_A2_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3560712, upper bound: 0.3299857
IS_A2_B1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3364038, upper bound: 0.3755019
IS_A2_B1_B2_A1_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3301412, upper bound: 0.3619098
IS_A2_B1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3364038, upper bound: 0.3755019
IS_A2_B1_B2_A1_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3301412, upper bound: 0.3619098
IS_A2_B2_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3787953, upper bound: 0.3408715
IS_A2_B2_A1_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3646159, upper bound: 0.3356805
IS_A2_B2_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3787953, upper bound: 0.3408715
IS_A2_B2_A1_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3646159, upper bound: 0.3356805
IS_A2_B2_A1_B1_A2_B1_B1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3372537, upper bound: 0.3550138
IS_A2_B2_A1_B1_A2_B1_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3311066, upper bound: 0.3321747
IS_A2_B2_A1_B1_A2_B2_B1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3372537, upper bound: 0.3550138
IS_A2_B2_A1_B1_A2_B2_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3311066, upper bound: 0.3321747
IS_A2_B2_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3391580, upper bound: 0.3762956
IS_A2_B2_A1_B2_B1_A1_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3339615, upper bound: 0.3642453
IS_A2_B2_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3391580, upper bound: 0.3762956
IS_A2_B2_A1_B2_B1_A2_B2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3339615, upper bound: 0.3642453
IS_A2_B2_A1_B2_B2_A1_A1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3518887, upper bound: 0.3360834
IS_A2_B2_A1_B2_B2_A1_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3305552, upper bound: 0.3301891
IS_A2_B2_A1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3518887, upper bound: 0.3360834
IS_A2_B2_A1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3305552, upper bound: 0.3301891
IS_A2_B2_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3742318, upper bound: 0.3404907
IS_A2_B2_A2_B1_A1_B1_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3630696, upper bound: 0.3353802
IS_A2_B2_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3742318, upper bound: 0.3404907
IS_A2_B2_A2_B1_A1_B2_A2, status: Status.VERIFIED, split count: 7, time: 5.67
Output dim: 7, lower bound: -0.3630696, upper bound: 0.3353802

## BFS IS instance: IS_A1_A1_B1_B1_A1_B1_B1

### Backsubstitution after applying IS history:
0: -0.1741697, 0.1182501, -0.1700701, 0.1146733, -0.2888430, 0.2883202
1: -0.1643215, 0.1231771, -0.1606942, 0.1183291, -0.2826506, 0.2838713
2: -0.0990503, 0.2159769, -0.0947295, 0.2114684, -0.3105187, 0.3107063
3: -0.0939293, 0.2683803, -0.0928630, 0.2614194, -0.3486822, 0.3546271
4: -0.1396105, 0.1573178, -0.1359306, 0.1530342, -0.2926446, 0.2932483
5: -0.1255827, 0.1981847, -0.1213784, 0.1931892, -0.3187719, 0.3195631
6: -0.1618320, 0.1492541, -0.1576256, 0.1444485, -0.3062805, 0.3068798
7: 0.5814726, 1.0688863, 0.5900311, 1.0680063, -0.4865337, 0.4788551
8: -0.1105620, 0.2326764, -0.1061597, 0.2277440, -0.3383060, 0.3388361
9: -0.1187423, 0.2279480, -0.1149327, 0.2230374, -0.3417796, 0.3428807

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3467707, upper bound: 0.3590345
time: 0.76 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 9

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_B1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3843095, upper bound: 0.3843095
time: 0.97 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B1_B1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3843095, upper bound: 0.3843095
time: 3.49 seconds

## BFS IS instance: IS_A1_A1_B1_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1700701, 0.1146733, -0.1674655, 0.1138410, -0.2839110, 0.2821388
1: -0.1606942, 0.1183291, -0.1575197, 0.1119215, -0.2726156, 0.2758488
2: -0.0947295, 0.2114684, -0.0910432, 0.2086956, -0.3034251, 0.3025116
3: -0.0928630, 0.2614194, -0.0890170, 0.2423350, -0.3283937, 0.3429804
4: -0.1359306, 0.1530342, -0.1337435, 0.1498526, -0.2857831, 0.2867777
5: -0.1213784, 0.1931892, -0.1177301, 0.1908531, -0.3122314, 0.3109193
6: -0.1576256, 0.1444485, -0.1544752, 0.1411408, -0.2987664, 0.2989237
7: 0.5900311, 1.0680063, 0.6142168, 1.0700700, -0.4800389, 0.4537895
8: -0.1061597, 0.2277440, -0.1036088, 0.2256446, -0.3318043, 0.3313528
9: -0.1149327, 0.2230374, -0.1114052, 0.2205204, -0.3354532, 0.3344426

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.48 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3704643, upper bound: 0.3311508
time: 0.90 seconds

## Relational analysis of IS_A1_A1_B1_B1_A1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B1_B1_A1_B2_A1_A2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3719955, upper bound: 0.3312236
time: 0.89 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1700701, 0.1146733, -0.1739336, 0.1183329, -0.2884030, 0.2886069
1: -0.1606942, 0.1183291, -0.1644406, 0.1231381, -0.2838323, 0.2827697
2: -0.0947295, 0.2114684, -0.0996165, 0.2161084, -0.3108379, 0.3110849
3: -0.0928630, 0.2614194, -0.1003036, 0.2683390, -0.3545214, 0.3551836
4: -0.1359306, 0.1530342, -0.1394941, 0.1572350, -0.2931656, 0.2925283
5: -0.1213784, 0.1931892, -0.1254569, 0.1979885, -0.3193669, 0.3186462
6: -0.1576256, 0.1444485, -0.1616902, 0.1494030, -0.3070287, 0.3061387
7: 0.5900311, 1.0680063, 0.5811852, 1.0756619, -0.4856308, 0.4868211
8: -0.1061597, 0.2277440, -0.1102747, 0.2328919, -0.3390517, 0.3380187
9: -0.1149327, 0.2230374, -0.1192082, 0.2278926, -0.3428254, 0.3422456

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.45 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3665798, upper bound: 0.3470915
time: 0.95 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_A1_B1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4013169, upper bound: 0.3879178
time: 1.10 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B1_A1_B2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.4013515, upper bound: 0.3877296
time: 1.04 seconds

## BFS IS instance: IS_A1_A1_B1_B2_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1700701, 0.1146733, -0.1672865, 0.1138846, -0.2839546, 0.2819598
1: -0.1606942, 0.1183291, -0.1575709, 0.1119591, -0.2726532, 0.2759000
2: -0.0947295, 0.2114684, -0.0914344, 0.2087702, -0.3034997, 0.3029028
3: -0.0928630, 0.2614194, -0.0957292, 0.2423134, -0.3285588, 0.3497921
4: -0.1359306, 0.1530342, -0.1336529, 0.1498031, -0.2857337, 0.2866871
5: -0.1213784, 0.1931892, -0.1176307, 0.1907236, -0.3121020, 0.3108200
6: -0.1576256, 0.1444485, -0.1543910, 0.1412252, -0.2988508, 0.2988395
7: 0.5900311, 1.0680063, 0.6140587, 1.0770576, -0.4870265, 0.4539475
8: -0.1061597, 0.2277440, -0.1033931, 0.2257753, -0.3319350, 0.3311372
9: -0.1149327, 0.2230374, -0.1117706, 0.2204498, -0.3353826, 0.3348080

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.54 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.19 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_A1_A1

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3770129, upper bound: 0.3349818
time: 0.95 seconds

## Relational analysis of IS_A1_A1_B1_B2_A1_B2_A1_A2

### Relational analysis result of IS_A1_A1_B1_B2_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3784415, upper bound: 0.3350533
time: 1.00 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B1_A1

### Backsubstitution after applying IS history:
0: -0.1714041, 0.1159641, -0.1739336, 0.1183329, -0.2897370, 0.2898977
1: -0.1620010, 0.1200114, -0.1644406, 0.1231381, -0.2851391, 0.2844519
2: -0.0965077, 0.2130706, -0.0996165, 0.2161084, -0.3126161, 0.3126871
3: -0.0975912, 0.2636106, -0.1003036, 0.2683390, -0.3586336, 0.3575000
4: -0.1371917, 0.1544925, -0.1394941, 0.1572350, -0.2944267, 0.2939866
5: -0.1227767, 0.1948895, -0.1254569, 0.1979885, -0.3207653, 0.3203465
6: -0.1590781, 0.1461449, -0.1616902, 0.1494030, -0.3084812, 0.3078351
7: 0.5871569, 1.0715237, 0.5811852, 1.0756619, -0.4885050, 0.4903384
8: -0.1075774, 0.2295461, -0.1102747, 0.2328919, -0.3404693, 0.3398207
9: -0.1164497, 0.2247165, -0.1192082, 0.2278926, -0.3443424, 0.3439246

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.50 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A1_B1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3435886, upper bound: 0.3336263
time: 0.93 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 9

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A1_B1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3846273, upper bound: 0.3834791
time: 0.99 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B1_A1_B2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3846273, upper bound: 0.3834791
time: 0.98 seconds

## BFS IS instance: IS_A1_A2_B2_B1_A1_B2_A1

### Backsubstitution after applying IS history:
0: -0.1714041, 0.1159641, -0.1672865, 0.1138846, -0.2852886, 0.2832506
1: -0.1620010, 0.1200114, -0.1575709, 0.1119591, -0.2739601, 0.2775823
2: -0.0965077, 0.2130706, -0.0914344, 0.2087702, -0.3052779, 0.3045051
3: -0.0975912, 0.2636106, -0.0957292, 0.2423134, -0.3326808, 0.3520881
4: -0.1371917, 0.1544925, -0.1336529, 0.1498031, -0.2869948, 0.2881454
5: -0.1227767, 0.1948895, -0.1176307, 0.1907236, -0.3135004, 0.3125203
6: -0.1590781, 0.1461449, -0.1543910, 0.1412252, -0.3003033, 0.3005359
7: 0.5871569, 1.0715237, 0.6140587, 1.0770576, -0.4899007, 0.4574649
8: -0.1075774, 0.2295461, -0.1033931, 0.2257753, -0.3333527, 0.3329392
9: -0.1164497, 0.2247165, -0.1117706, 0.2204498, -0.3368995, 0.3364871

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=11, inp2_unstable=12, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.56 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.17 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1_A1

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3719963, upper bound: 0.3336799
time: 1.01 seconds

## Relational analysis of IS_A1_A2_B2_B1_A1_B2_A1_A2

### Relational analysis result of IS_A1_A2_B2_B1_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3732946, upper bound: 0.3338347
time: 0.93 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1739336, 0.1183329, -0.1700701, 0.1146733, -0.2886069, 0.2884030
1: -0.1644406, 0.1231381, -0.1606942, 0.1183291, -0.2827697, 0.2838323
2: -0.0996165, 0.2161084, -0.0947295, 0.2114684, -0.3110849, 0.3108379
3: -0.1003036, 0.2683390, -0.0928630, 0.2614194, -0.3551837, 0.3545214
4: -0.1394941, 0.1572350, -0.1359306, 0.1530342, -0.2925283, 0.2931656
5: -0.1254569, 0.1979885, -0.1213784, 0.1931892, -0.3186462, 0.3193669
6: -0.1616902, 0.1494030, -0.1576256, 0.1444485, -0.3061387, 0.3070287
7: 0.5811852, 1.0756619, 0.5900311, 1.0680063, -0.4868211, 0.4856308
8: -0.1102747, 0.2328919, -0.1061597, 0.2277440, -0.3380187, 0.3390517
9: -0.1192082, 0.2278926, -0.1149327, 0.2230374, -0.3422456, 0.3428254

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.62 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 86
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 86
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 153
type: B, layer: 1, pos: 153
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.18 seconds

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.3470915, upper bound: 0.3665798
time: 0.90 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1_A1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3879178, upper bound: 0.4013169
time: 1.13 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A1_B1_A2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3877296, upper bound: 0.4013515
time: 1.05 seconds

## BFS IS instance: IS_A2_B1_B1_A1_B1_A2_B1

### Backsubstitution after applying IS history:
0: -0.1672865, 0.1138846, -0.1700701, 0.1146733, -0.2819598, 0.2839546
1: -0.1575709, 0.1119591, -0.1606942, 0.1183291, -0.2759000, 0.2726532
2: -0.0914344, 0.2087702, -0.0947295, 0.2114684, -0.3029028, 0.3034997
3: -0.0957292, 0.2423134, -0.0928630, 0.2614194, -0.3497921, 0.3285588
4: -0.1336529, 0.1498031, -0.1359306, 0.1530342, -0.2866871, 0.2857337
5: -0.1176307, 0.1907236, -0.1213784, 0.1931892, -0.3108200, 0.3121020
6: -0.1543910, 0.1412252, -0.1576256, 0.1444485, -0.2988395, 0.2988508
7: 0.6140587, 1.0770576, 0.5900311, 1.0680063, -0.4539475, 0.4870265
8: -0.1033931, 0.2257753, -0.1061597, 0.2277440, -0.3311372, 0.3319350
9: -0.1117706, 0.2204498, -0.1149327, 0.2230374, -0.3348080, 0.3353826

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=17, inp2_unstable=18, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.52 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 219
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 210
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 132
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 132
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 72
type: B, layer: 1, pos: 72
type: B, layer: 1, pos: 57
type: A, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 86

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1_B1

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3349818, upper bound: 0.3770129
time: 0.92 seconds

## Relational analysis of IS_A2_B1_B1_A1_B1_A2_B1_B2

### Relational analysis result of IS_A2_B1_B1_A1_B1_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.3350533, upper bound: 0.3784415
time: 0.97 seconds

## BFS IS instance: IS_A2_B1_B2_A1_B1_A1_B1

### Backsubstitution after applying IS history:
0: -0.1739336, 0.1183329, -0.1714041, 0.1159641, -0.2898977, 0.2897370
1: -0.1644406, 0.1231381, -0.1620010, 0.1200114, -0.2844519, 0.2851391
2: -0.0996165, 0.2161084, -0.0965077, 0.2130706, -0.3126871, 0.3126161
3: -0.1003036, 0.2683390, -0.0975912, 0.2636106, -0.3575000, 0.3586336
4: -0.1394941, 0.1572350, -0.1371917, 0.1544925, -0.2939866, 0.2944267
5: -0.1254569, 0.1979885, -0.1227767, 0.1948895, -0.3203465, 0.3207653
6: -0.1616902, 0.1494030, -0.1590781, 0.1461449, -0.3078351, 0.3084812
7: 0.5811852, 1.0756619, 0.5871569, 1.0715237, -0.4903384, 0.4885050
8: -0.1102747, 0.2328919, -0.1075774, 0.2295461, -0.3398207, 0.3404693
9: -0.1192082, 0.2278926, -0.1164497, 0.2247165, -0.3439246, 0.3443424

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=12, inp2_unstable=11, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=18, inp2_unstable=17, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=2, inp2_unstable=2, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=30, inp2_unstable=30, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=9, inp2_unstable=9, delta_unstable=10

Time for backsubstitution: 1.49 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 93
type: A, layer: 1, pos: 93
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 86
type: B, layer: 1, pos: 86
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 219
type: B, layer: 1, pos: 219
type: B, layer: 1, pos: 210
type: A, layer: 1, pos: 210
type: B, layer: 1, pos: 153
type: A, layer: 1, pos: 153
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 132
type: A, layer: 1, pos: 132
type: B, layer: 1, pos: 72
type: A, layer: 1, pos: 72
type: A, layer: 1, pos: 57
type: B, layer: 1, pos: 57

Time for candidate selection: 0.16 seconds

### Candidate
type: B, layer: 1, pos: 93

## Relational analysis of IS_A2_B1_B2_A1_B1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 3.55 + 598.00 = 601.55 seconds
