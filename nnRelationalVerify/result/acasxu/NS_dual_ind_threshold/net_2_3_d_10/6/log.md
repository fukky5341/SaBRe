## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_2_3.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 4905.232506984402


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1477.5816650, 5163.0268555, -1477.5816650, 5163.0268555, -6640.6083984, 6640.6083984)
1: (-1482.7659912, 3281.7324219, -1482.7659912, 3281.7324219, -4764.4985352, 4764.4985352)
2: (-1346.8199463, 3223.7636719, -1346.8199463, 3223.7636719, -4570.5830078, 4570.5834961)
3: (-1611.3056641, 3905.3154297, -1611.3056641, 3905.3154297, -5516.6210938, 5516.6210938)
4: (-1867.7043457, 3542.1357422, -1867.7043457, 3542.1357422, -5409.8398438, 5409.8398438)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.73 + 2.36 = 4.08 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -4905.2815598, upper bound: 4905.2815598

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 40

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2812866, upper bound: 4905.2812684
time: 0.87 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2815598, upper bound: 4905.2815598
time: 0.88 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 1.89 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 3, lower bound: -4905.2812866, upper bound: 4905.2812684
NS_A2, status: Status.UNKNOWN, split count: 1, time: 1.89
Output dim: 3, lower bound: -4905.2815598, upper bound: 4905.2815598

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1478.5439453, 5165.9506836, -1453.2752686, 5078.3696289, -6556.9130859, 6619.2260742
1: -1482.1302490, 3284.5915527, -1458.4094238, 3227.7607422, -4709.8911133, 4743.0009766
2: -1347.9916992, 3225.2797852, -1324.6729736, 3170.9919434, -4518.9833984, 4549.9526367
3: -1612.4108887, 3908.9294434, -1584.9038086, 3841.0324707, -5453.4433594, 5493.8330078
4: -1869.6408691, 3544.5710449, -1836.9908447, 3484.1589355, -5353.7993164, 5381.5610352

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2812728, upper bound: 4905.2811984
time: 1.04 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2812728, upper bound: 4905.2812523
time: 0.97 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1463.6452637, 5114.4799805, -1468.3205566, 5130.7109375, -6594.3564453, 6582.8002930
1: -1468.6223145, 3250.8281250, -1473.3657227, 3261.1816406, -4729.8037109, 4724.1938477
2: -1334.1149902, 3193.4206543, -1338.3669434, 3203.5822754, -4537.6972656, 4531.7875977
3: -1596.0220947, 3868.4804688, -1601.1511230, 3880.8107910, -5476.8330078, 5469.6318359
4: -1850.1325684, 3508.7770996, -1856.0104980, 3519.9553223, -5370.0874023, 5364.7871094

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 24

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2815101, upper bound: 4905.2815492
time: 0.99 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2815101, upper bound: 4905.2815101
time: 0.81 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 3.50 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 3, lower bound: -4905.2812728, upper bound: 4905.2811984
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 3, lower bound: -4905.2812728, upper bound: 4905.2812523
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 3, lower bound: -4905.2815101, upper bound: 4905.2815492
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 3.50
Output dim: 3, lower bound: -4905.2815101, upper bound: 4905.2815101

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1458.9711914, 5097.2929688, -1426.2336426, 4983.7041016, -6442.6752930, 6523.5263672
1: -1462.1428223, 3241.0180664, -1430.9437256, 3167.6848145, -4629.8276367, 4671.9619141
2: -1329.8463135, 3182.4943848, -1299.6687012, 3112.0070801, -4441.8535156, 4482.1630859
3: -1590.8281250, 3856.8706055, -1555.1535645, 3769.2387695, -5360.0654297, 5412.0244141
4: -1844.6551514, 3497.5771484, -1802.5692139, 3419.3183594, -5263.9736328, 5300.1464844

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2790993, upper bound: 4905.2792265
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2781389, upper bound: 4905.2774718
time: 0.97 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1460.2591553, 5103.2895508, -1513.6578369, 5293.1093750, -6753.3681641, 6616.9472656
1: -1463.7761230, 3244.5507812, -1517.3369141, 3363.9250488, -4827.7011719, 4761.8872070
2: -1331.6291504, 3186.0551758, -1382.1737061, 3303.6687012, -4635.2968750, 4568.2290039
3: -1592.4752197, 3861.3576660, -1651.0745850, 4004.5974121, -5597.0722656, 5512.4321289
4: -1846.9908447, 3501.1462402, -1915.9984131, 3629.0698242, -5476.0605469, 5417.1445312

Time for backsubstitution: 1.58 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2790993, upper bound: 4905.2793594
time: 0.85 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2781389, upper bound: 4905.2776646
time: 0.90 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1448.8193359, 5062.8081055, -1446.8056641, 5055.7851562, -6504.6044922, 6509.6137695
1: -1453.5623779, 3217.9138184, -1451.5095215, 3213.4655762, -4667.0268555, 4669.4233398
2: -1320.4505615, 3161.0961914, -1318.5622559, 3156.7124023, -4477.1621094, 4479.6582031
3: -1579.7333984, 3829.2221680, -1577.5126953, 3823.8925781, -5403.6259766, 5406.7343750
4: -1831.3428955, 3473.2409668, -1828.7762451, 3468.4023438, -5299.7446289, 5302.0170898

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2793442, upper bound: 4905.2799368
time: 1.03 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2785285, upper bound: 4905.2785707
time: 0.99 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1442.3289795, 5040.6606445, -1527.4787598, 5340.6401367, -6782.9692383, 6568.1391602
1: -1447.1695557, 3204.0327148, -1530.9068604, 3394.4782715, -4841.6474609, 4734.9389648
2: -1314.9284668, 3147.4418945, -1394.5510254, 3333.2822266, -4648.2109375, 4541.9931641
3: -1572.7001953, 3812.7358398, -1665.8281250, 4040.9150391, -5613.6132812, 5478.5639648
4: -1823.5295410, 3457.9389648, -1933.2259521, 3661.6108398, -5485.1406250, 5391.1645508

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 19

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2793442, upper bound: 4905.2798252
time: 1.14 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2785285, upper bound: 4905.2785285
time: 1.07 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.95 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -4905.2790993, upper bound: 4905.2792265
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -4905.2781389, upper bound: 4905.2774718
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -4905.2790993, upper bound: 4905.2793594
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -4905.2781389, upper bound: 4905.2776646
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -4905.2793442, upper bound: 4905.2799368
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -4905.2785285, upper bound: 4905.2785707
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -4905.2793442, upper bound: 4905.2798252
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.95
Output dim: 3, lower bound: -4905.2785285, upper bound: 4905.2785285

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1431.2145996, 5000.7695312, -1413.2911377, 4938.5893555, -6369.8037109, 6414.0605469
1: -1434.4062500, 3180.1330566, -1417.9755859, 3139.2961426, -4573.7021484, 4598.1083984
2: -1305.2937012, 3122.6862793, -1288.2114258, 3084.0869141, -4389.3808594, 4410.8974609
3: -1560.6500244, 3784.4821777, -1541.0437012, 3735.4194336, -5296.0693359, 5325.5249023
4: -1810.3568115, 3431.5124512, -1786.5430908, 3388.4941406, -5198.8510742, 5218.0546875

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2785076, upper bound: 4905.2786051
time: 0.96 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2789673, upper bound: 4905.2789070
time: 1.03 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1465.0072021, 5115.9555664, -1405.1363525, 4910.7036133, -6375.7104492, 6521.0917969
1: -1468.6540527, 3254.0593262, -1410.0144043, 3121.5512695, -4590.2050781, 4664.0732422
2: -1336.1041260, 3196.1123047, -1281.1182861, 3066.7648926, -4402.8691406, 4477.2304688
3: -1597.9072266, 3872.7521973, -1532.4453125, 3714.5122070, -5312.4194336, 5405.1962891
4: -1852.2634277, 3511.7729492, -1776.5910645, 3369.4033203, -5221.6669922, 5288.3642578

Time for backsubstitution: 1.59 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2774658, upper bound: 4905.2768485
time: 1.40 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2780232, upper bound: 4905.2774153
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1432.4849854, 5006.6508789, -1499.6398926, 5244.2036133, -6676.6884766, 6506.2900391
1: -1436.0541992, 3183.5917969, -1503.3176270, 3333.1708984, -4769.2250977, 4686.9082031
2: -1307.0750732, 3126.1477051, -1369.8045654, 3273.4440918, -4580.5190430, 4495.9521484
3: -1562.3121338, 3788.9191895, -1635.8286133, 3967.9567871, -5530.2690430, 5424.7475586
4: -1812.6680908, 3435.0104980, -1898.6923828, 3595.6906738, -5408.3588867, 5333.7031250

Time for backsubstitution: 1.60 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2781078, upper bound: 4905.2786963
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2789673, upper bound: 4905.2791530
time: 1.16 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1467.5699463, 5126.1469727, -1496.0335693, 5231.9536133, -6699.5234375, 6622.1806641
1: -1471.5390625, 3260.3212891, -1499.7926025, 3325.2441406, -4796.7817383, 4760.1137695
2: -1338.9599609, 3202.3637695, -1366.4809570, 3265.7158203, -4604.6757812, 4568.8447266
3: -1600.9387207, 3880.3972168, -1631.9630127, 3958.6550293, -5559.5937500, 5512.3598633
4: -1856.0897217, 3518.3139648, -1894.0780029, 3587.2263184, -5443.3159180, 5412.3920898

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2772190, upper bound: 4905.2768573
time: 1.11 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2780232, upper bound: 4905.2776188
time: 0.88 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1422.6339111, 4971.3139648, -1433.6733398, 5010.0400391, -6432.6738281, 6404.9873047
1: -1427.3393555, 3160.4160156, -1438.3488770, 3184.7163086, -4612.0551758, 4598.7646484
2: -1297.2520752, 3104.4948730, -1306.9554443, 3128.4360352, -4425.6875000, 4411.4487305
3: -1551.2055664, 3760.6970215, -1563.2072754, 3789.6003418, -5340.8056641, 5323.9042969
4: -1798.8748779, 3410.8115234, -1812.5450439, 3437.1838379, -5236.0585938, 5223.3559570

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2787750, upper bound: 4905.2793609
time: 1.11 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2791157, upper bound: 4905.2793702
time: 1.27 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1448.5588379, 5060.2504883, -1425.6242676, 4982.3208008, -6430.8789062, 6485.8750000
1: -1454.0557861, 3217.0495605, -1430.4822998, 3167.0515137, -4621.1069336, 4647.5317383
2: -1321.0146484, 3161.2766113, -1299.8773193, 3111.1794434, -4432.1933594, 4461.1538086
3: -1580.5655518, 3828.9799805, -1554.6967773, 3768.8217773, -5349.3872070, 5383.6757812
4: -1831.1948242, 3472.7058105, -1802.5968018, 3418.1962891, -5249.3911133, 5275.3027344

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2779548, upper bound: 4905.2783603
time: 0.90 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2783541, upper bound: 4905.2783824
time: 1.28 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1416.4309082, 4950.0986328, -1513.4528809, 5291.7216797, -6708.1523438, 6463.5517578
1: -1421.2767334, 3147.0942383, -1516.8703613, 3363.7165527, -4784.9931641, 4663.9638672
2: -1291.9862061, 3091.3815918, -1382.1843262, 3303.0434570, -4595.0297852, 4473.5659180
3: -1544.5322266, 3744.9501953, -1650.5675049, 4004.2631836, -5548.7954102, 5395.5175781
4: -1791.3950195, 3396.1442871, -1915.9207764, 3628.2062988, -5419.6000977, 5312.0649414

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2781180, upper bound: 4905.2792630
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2791157, upper bound: 4905.2793192
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1443.4885254, 5043.1796875, -1509.6578369, 5278.8442383, -6722.3330078, 6552.8374023
1: -1449.0753174, 3206.4113770, -1513.1914062, 3355.3479004, -4804.4228516, 4719.6025391
2: -1316.8161621, 3150.7978516, -1378.7011719, 3294.9235840, -4611.7392578, 4529.4990234
3: -1575.0871582, 3816.3015137, -1646.5279541, 3994.4697266, -5569.5566406, 5462.8295898
4: -1825.2275391, 3460.8630371, -1911.0662842, 3619.3291016, -5444.5566406, 5371.9287109

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 2

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2773342, upper bound: 4905.2782656
time: 0.99 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2783541, upper bound: 4905.2783541
time: 1.54 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 4.28 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2785076, upper bound: 4905.2786051
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2789673, upper bound: 4905.2789070
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2774658, upper bound: 4905.2768485
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2780232, upper bound: 4905.2774153
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2781078, upper bound: 4905.2786963
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2789673, upper bound: 4905.2791530
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2772190, upper bound: 4905.2768573
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2780232, upper bound: 4905.2776188
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2787750, upper bound: 4905.2793609
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2791157, upper bound: 4905.2793702
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2779548, upper bound: 4905.2783603
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2783541, upper bound: 4905.2783824
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2781180, upper bound: 4905.2792630
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2791157, upper bound: 4905.2793192
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2773342, upper bound: 4905.2782656
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 4.28
Output dim: 3, lower bound: -4905.2783541, upper bound: 4905.2783541

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1414.6801758, 4942.8798828, -1367.5714111, 4778.7441406, -6193.4238281, 6310.4506836
1: -1417.5759277, 3143.3403320, -1371.2575684, 3036.9218750, -4454.4980469, 4514.5976562
2: -1290.2567139, 3086.3645020, -1244.9820557, 2983.7282715, -4273.9848633, 4331.3466797
3: -1542.5697021, 3740.6401367, -1490.3631592, 3613.0310059, -5155.6005859, 5231.0029297
4: -1789.5491943, 3391.6982422, -1727.4626465, 3277.8295898, -5067.3784180, 5119.1611328

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2677503, upper bound: 4905.2694307
time: 0.87 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2784177, upper bound: 4905.2785532
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1426.8847656, 4985.7114258, -1407.7028809, 4919.2197266, -6346.1044922, 6393.4140625
1: -1430.1437988, 3170.5893555, -1412.4578857, 3126.8076172, -4556.9511719, 4583.0473633
2: -1301.4748535, 3113.2260742, -1283.2221680, 3071.8439941, -4373.3183594, 4396.4482422
3: -1556.0234375, 3773.2434082, -1535.0451660, 3720.7199707, -5276.7426758, 5308.2885742
4: -1805.0350342, 3421.1765137, -1779.5831299, 3375.1054688, -5180.1406250, 5200.7583008

Time for backsubstitution: 1.61 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2780654, upper bound: 4905.2782954
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2788224, upper bound: 4905.2788170
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1447.9796143, 5056.5102539, -1358.6002197, 4748.2700195, -6196.2495117, 6415.1103516
1: -1451.3428955, 3216.2263184, -1362.4394531, 3017.4604492, -4468.8032227, 4578.6650391
2: -1320.6901855, 3158.7536621, -1237.2197266, 2964.7656250, -4285.4550781, 4395.9731445
3: -1579.2841797, 3827.6940918, -1480.8223877, 3590.0988770, -5169.3828125, 5308.5166016
4: -1830.9285889, 3470.7800293, -1716.6003418, 3256.8715820, -5087.8002930, 5187.3798828

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2700034, upper bound: 4905.2680808
time: 1.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2772855, upper bound: 4905.2763860
time: 0.91 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1461.1280518, 5102.4433594, -1399.5345459, 4891.3164062, -6352.4443359, 6501.9770508
1: -1464.8164062, 3245.5117188, -1404.4797363, 3109.0351562, -4573.8515625, 4649.9912109
2: -1332.6656494, 3187.6367188, -1276.1157227, 3054.5061035, -4387.1718750, 4463.7524414
3: -1593.7459717, 3862.6765137, -1526.4299316, 3699.7929688, -5293.5380859, 5389.1064453
4: -1847.4709473, 3502.5070801, -1769.6191406, 3355.9902344, -5203.4609375, 5272.1259766

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2719060, upper bound: 4905.2697547
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2779230, upper bound: 4905.2773079
time: 1.08 seconds

## BFS NS instance: NS_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1414.4307861, 4943.7832031, -1440.3448486, 5041.0654297, -6455.4960938, 6384.1279297
1: -1417.6910400, 3143.5263672, -1443.6629639, 3202.3098145, -4620.0009766, 4587.1879883
2: -1290.7601318, 3086.5961914, -1315.1190186, 3146.3027344, -4437.0625000, 4401.7153320
3: -1542.5700684, 3741.2770996, -1570.3062744, 3811.8039551, -5354.3735352, 5311.5815430
4: -1790.1151123, 3391.5878906, -1824.1357422, 3455.1411133, -5245.2563477, 5215.7231445

Time for backsubstitution: 1.62 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2778705, upper bound: 4905.2781648
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2778639, upper bound: 4905.2781870
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1428.4193115, 4992.4877930, -1492.2182617, 5217.7465820, -6646.1660156, 6484.7055664
1: -1432.0386963, 3174.6210938, -1495.9033203, 3316.5617676, -4748.6000977, 4670.5234375
2: -1303.4752197, 3117.2509766, -1363.1866455, 3256.8034668, -4560.2783203, 4480.4375000
3: -1557.9602051, 3778.3515625, -1627.8405762, 3948.3134766, -5506.2734375, 5406.1923828
4: -1807.6574707, 3425.2919922, -1889.4146729, 3577.5654297, -5385.2226562, 5314.7065430

Time for backsubstitution: 1.63 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2780654, upper bound: 4905.2785621
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2788224, upper bound: 4905.2790667
time: 0.86 seconds

## BFS NS instance: NS_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1449.3975830, 5062.9531250, -1436.0766602, 5026.4516602, -6475.8491211, 6499.0297852
1: -1453.0800781, 3220.0581055, -1439.4432373, 3192.9101562, -4645.9902344, 4659.4995117
2: -1322.5991211, 3162.5676270, -1311.1967773, 3137.1149902, -4459.7138672, 4473.7646484
3: -1581.0002441, 3832.5029297, -1565.6965332, 3800.7258301, -5381.7260742, 5398.1992188
4: -1833.4534912, 3474.5908203, -1818.6727295, 3445.0795898, -5278.5332031, 5293.2631836

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2699910, upper bound: 4905.2679701
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2769453, upper bound: 4905.2764217
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1463.6984863, 5112.6606445, -1488.4221191, 5204.8281250, -6668.5263672, 6601.0820312
1: -1467.7069092, 3251.7875977, -1492.1750488, 3308.2216797, -4775.9282227, 4743.9614258
2: -1335.5241699, 3193.9018555, -1359.6951904, 3248.6691895, -4584.1928711, 4553.5971680
3: -1596.7828369, 3870.3435059, -1623.7561035, 3938.5141602, -5535.2968750, 5494.0996094
4: -1851.3021240, 3509.0625000, -1884.5642090, 3568.6523438, -5419.9536133, 5393.6269531

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2723998, upper bound: 4905.2701763
time: 1.07 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2779230, upper bound: 4905.2775893
time: 0.92 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1403.1601562, 4902.9448242, -1387.6383057, 4849.2954102, -6252.4555664, 6290.5825195
1: -1407.7675781, 3117.2910156, -1391.3127441, 3081.6020508, -4489.3696289, 4508.6035156
2: -1279.7969971, 3061.8059082, -1263.3691406, 3027.4531250, -4307.2500000, 4325.1748047
3: -1530.0747070, 3709.3552246, -1512.0938721, 3666.3181152, -5196.3925781, 5221.4492188
4: -1774.5471191, 3364.0410156, -1753.0694580, 3325.7490234, -5100.2958984, 5117.1103516

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2779356, upper bound: 4905.2784234
time: 0.96 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2786719, upper bound: 4905.2793362
time: 0.90 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1417.0668945, 4951.9145508, -1425.8177490, 4982.7402344, -6399.8061523, 6377.7319336
1: -1421.8277588, 3147.9553223, -1430.5560303, 3167.1809082, -4589.0087891, 4578.5107422
2: -1292.2795410, 3092.2595215, -1299.9378662, 3111.2331543, -4403.5126953, 4392.1972656
3: -1545.2281494, 3746.0053711, -1554.7532959, 3768.9082031, -5314.1357422, 5300.7587891
4: -1791.9407959, 3397.4624023, -1802.7714844, 3418.3859863, -5210.3266602, 5200.2338867

Time for backsubstitution: 1.64 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2785697, upper bound: 4905.2792934
time: 1.34 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2789912, upper bound: 4905.2793422
time: 0.96 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1428.9478760, 4991.6069336, -1378.8164062, 4819.1342773, -6248.0815430, 6370.4233398
1: -1434.3254395, 3173.7233887, -1382.6523438, 3062.3647461, -4496.6899414, 4556.3745117
2: -1303.5020752, 3118.4006348, -1255.6893311, 3008.6713867, -4312.1733398, 4374.0898438
3: -1559.2739258, 3777.3601074, -1502.7255859, 3643.6804199, -5202.9541016, 5280.0859375
4: -1806.8239746, 3425.7119141, -1742.2834473, 3305.0478516, -5111.8710938, 5167.9936523

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2684386, upper bound: 4905.2676957
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2778438, upper bound: 4905.2782621
time: 0.81 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1443.0594482, 5041.1235352, -1417.7241211, 4954.8725586, -6397.9321289, 6458.8476562
1: -1448.5975342, 3204.7539062, -1422.6375732, 3149.4147949, -4598.0122070, 4627.3916016
2: -1316.1042480, 3149.2114258, -1292.8165283, 3093.8803711, -4409.9843750, 4442.0278320
3: -1574.6490479, 3814.4807129, -1546.1934814, 3748.0126953, -5322.6616211, 5360.6743164
4: -1824.3481445, 3459.5368652, -1792.7666016, 3399.2893066, -5223.6367188, 5252.3027344

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2690486, upper bound: 4905.2683033
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2783146, upper bound: 4905.2783532
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1396.5061035, 4880.1367188, -1454.9300537, 5091.3740234, -6487.8798828, 6335.0668945
1: -1401.2520752, 3102.9509277, -1457.9921875, 3234.6999512, -4635.9521484, 4560.9433594
2: -1274.0885010, 3047.6867676, -1328.1643066, 3177.7709961, -4451.8593750, 4375.8505859
3: -1522.8990479, 3692.4001465, -1585.8209229, 3850.1367188, -5373.0356445, 5278.2211914
4: -1766.4747314, 3348.2885742, -1842.3183594, 3489.5603027, -5256.0351562, 5190.6069336

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2778736, upper bound: 4905.2784252
time: 1.20 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2778730, upper bound: 4905.2789062
time: 1.06 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1411.8328857, 4934.0336914, -1504.0767822, 5258.5249023, -6670.3579102, 6438.1103516
1: -1416.7531738, 3136.7822266, -1507.5202637, 3342.8173828, -4759.5703125, 4644.3027344
2: -1287.9017334, 3081.2321777, -1373.8621826, 3282.2055664, -4570.1074219, 4455.0942383
3: -1539.6168213, 3732.8012695, -1640.4764404, 3979.5080566, -5519.1240234, 5373.2773438
4: -1785.6818848, 3385.0930176, -1904.2678223, 3605.4812012, -5391.1621094, 5289.3603516

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2791034, upper bound: 4905.2786734
time: 0.96 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2791034, upper bound: 4905.2790088
time: 0.92 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1423.3564453, 4972.6718750, -1450.4344482, 5075.9931641, -6499.3496094, 6423.1064453
1: -1428.8068848, 3161.8789062, -1453.5800781, 3224.7890625, -4653.5957031, 4615.4584961
2: -1298.7724609, 3106.7475586, -1324.0512695, 3168.1132812, -4466.8857422, 4430.7988281
3: -1553.1904297, 3763.2609863, -1581.0023193, 3838.4841309, -5391.6733398, 5344.2631836
4: -1800.1301270, 3412.5883789, -1836.5759277, 3479.0024414, -5279.1328125, 5249.1640625

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2679290, upper bound: 4905.2675109
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2770213, upper bound: 4905.2781094
time: 1.01 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1439.3406982, 5028.6484375, -1500.2198486, 5245.4487305, -6684.7890625, 6528.8681641
1: -1444.9875488, 3197.0888672, -1503.7502441, 3334.3310547, -4779.3183594, 4700.8383789
2: -1313.1138916, 3141.6225586, -1370.3212891, 3273.9621582, -4587.0756836, 4511.9438477
3: -1570.6485596, 3805.3034668, -1636.3433838, 3969.5593262, -5540.2080078, 5441.6469727
4: -1820.0455322, 3450.8806152, -1899.3399658, 3596.4602051, -5416.5058594, 5350.2207031

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2690486, upper bound: 4905.2682881
time: 1.79 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2783146, upper bound: 4905.2783146
time: 1.10 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 4.70 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2677503, upper bound: 4905.2694307
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2784177, upper bound: 4905.2785532
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2780654, upper bound: 4905.2782954
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2788224, upper bound: 4905.2788170
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2700034, upper bound: 4905.2680808
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2772855, upper bound: 4905.2763860
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2719060, upper bound: 4905.2697547
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2779230, upper bound: 4905.2773079
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2778705, upper bound: 4905.2781648
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2778639, upper bound: 4905.2781870
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2780654, upper bound: 4905.2785621
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2788224, upper bound: 4905.2790667
NS_A1_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2699910, upper bound: 4905.2679701
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2769453, upper bound: 4905.2764217
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2723998, upper bound: 4905.2701763
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2779230, upper bound: 4905.2775893
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2779356, upper bound: 4905.2784234
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2786719, upper bound: 4905.2793362
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2785697, upper bound: 4905.2792934
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2789912, upper bound: 4905.2793422
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2684386, upper bound: 4905.2676957
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2778438, upper bound: 4905.2782621
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2690486, upper bound: 4905.2683033
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2783146, upper bound: 4905.2783532
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2778736, upper bound: 4905.2784252
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2778730, upper bound: 4905.2789062
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2791034, upper bound: 4905.2786734
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2791034, upper bound: 4905.2790088
NS_A2_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2679290, upper bound: 4905.2675109
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2770213, upper bound: 4905.2781094
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2690486, upper bound: 4905.2682881
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 4.70
Output dim: 3, lower bound: -4905.2783146, upper bound: 4905.2783146

## BFS NS instance: NS_A1_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1390.8065186, 4859.8442383, -1347.1907959, 4708.3027344, -6099.1093750, 6207.0351562
1: -1393.6210938, 3090.4182129, -1350.8421631, 2992.1069336, -4385.7275391, 4441.2592773
2: -1268.8743896, 3034.4016113, -1226.7978516, 2939.7404785, -4208.6147461, 4261.1992188
3: -1516.6495361, 3677.8715820, -1468.1481934, 3559.7280273, -5076.3774414, 5146.0195312
4: -1759.8172607, 3334.5952148, -1702.2305908, 3229.3022461, -4989.1196289, 5036.8256836

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2663945, upper bound: 4905.2666781
time: 1.09 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2669861, upper bound: 4905.2669381
time: 0.81 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1413.6224365, 4939.5654297, -1351.3087158, 4722.0654297, -6135.6879883, 6290.8740234
1: -1417.4603271, 3141.3212891, -1355.4127197, 3000.5092773, -4417.9692383, 4496.7338867
2: -1290.8226318, 3083.9541016, -1230.6624756, 2947.8427734, -4238.6650391, 4314.6166992
3: -1542.7612305, 3739.0195312, -1473.1506348, 3570.2617188, -5113.0224609, 5212.1699219
4: -1789.4925537, 3389.0024414, -1707.3088379, 3238.4006348, -5027.8930664, 5096.3110352

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2663045, upper bound: 4905.2671629
time: 0.90 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2782680, upper bound: 4905.2779902
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1402.9161377, 4902.4033203, -1388.3461914, 4852.1674805, -6255.0834961, 6290.7495117
1: -1406.1177979, 3117.4821777, -1393.0961914, 3084.3024902, -4490.4199219, 4510.5781250
2: -1280.0562744, 3061.0393066, -1266.0847168, 3030.0710449, -4310.1274414, 4327.1240234
3: -1530.0169678, 3710.2504883, -1514.0518799, 3670.1567383, -5200.1738281, 5224.3022461
4: -1775.2371826, 3363.8173828, -1755.6655273, 3329.1271973, -5104.3642578, 5119.4829102

Time for backsubstitution: 1.65 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A1_B1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2672286, upper bound: 4905.2681453
time: 0.76 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2779380, upper bound: 4905.2778999
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1424.8710938, 4979.2065430, -1390.5616455, 4859.5634766, -6284.4340820, 6369.7680664
1: -1429.1292725, 3166.4584961, -1395.7401123, 3088.5087891, -4517.6381836, 4562.1987305
2: -1301.2293701, 3108.7290039, -1268.1557617, 3034.0927734, -4335.3222656, 4376.8847656
3: -1555.2384033, 3769.1706543, -1516.8872070, 3675.7260742, -5230.9643555, 5286.0566406
4: -1803.8448486, 3416.2089844, -1758.3864746, 3333.6269531, -5137.4716797, 5174.5957031

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B1_A1_B2_A2_B1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2674484, upper bound: 4905.2682409
time: 0.97 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2787408, upper bound: 4905.2783029
time: 0.82 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1426.6031494, 4981.6933594, -1345.3010254, 4701.7763672, -6128.3793945, 6326.9941406
1: -1429.6030273, 3168.6840820, -1349.0142822, 2987.8730469, -4417.4760742, 4517.6982422
2: -1300.9510498, 3112.1379395, -1224.9136963, 2935.8034668, -4236.7539062, 4337.0502930
3: -1555.8273926, 3770.9431152, -1466.2402344, 3554.7619629, -5110.5888672, 5237.1835938
4: -1803.7315674, 3419.6076660, -1699.6809082, 3225.0695801, -5028.7993164, 5119.2885742

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2685644, upper bound: 4905.2675531
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2699893, upper bound: 4905.2680309
time: 0.95 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1451.9149170, 5073.2153320, -1348.0212402, 4710.7695312, -6162.6845703, 6421.2363281
1: -1456.1729736, 3226.0290527, -1351.7829590, 2993.7612305, -4449.9340820, 4577.8120117
2: -1327.0311279, 3169.4753418, -1227.4216309, 2941.4333496, -4268.4638672, 4396.8964844
3: -1585.0557861, 3840.6364746, -1469.3161621, 3561.8701172, -5146.9257812, 5309.9526367
4: -1838.9229736, 3482.8115234, -1703.0205078, 3231.3811035, -5070.3041992, 5185.8310547

Time for backsubstitution: 1.66 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2712149, upper bound: 4905.2700290
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2772618, upper bound: 4905.2763054
time: 1.10 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1439.8996582, 5028.1679688, -1386.1004639, 4844.4335938, -6284.3330078, 6414.2685547
1: -1443.2084961, 3198.2961426, -1390.9127197, 3079.2355957, -4522.4443359, 4589.2075195
2: -1313.0714111, 3141.3154297, -1263.7833252, 3025.3129883, -4338.3842773, 4405.0986328
3: -1570.4365234, 3806.3195801, -1511.7094727, 3664.2021484, -5234.6381836, 5318.0292969
4: -1820.4805908, 3451.6528320, -1752.6217041, 3323.9301758, -5144.4091797, 5204.2744141

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2716380, upper bound: 4905.2691485
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2716431, upper bound: 4905.2692447
time: 0.84 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1465.8137207, 5121.6035156, -1388.9790039, 4853.8627930, -6319.6767578, 6510.5825195
1: -1470.3530273, 3256.8078613, -1393.8419189, 3085.4423828, -4555.7944336, 4650.6499023
2: -1339.5992432, 3199.8913574, -1266.4290771, 3031.2509766, -4370.8500977, 4466.3203125
3: -1600.3692627, 3877.3581543, -1514.9927979, 3671.6936035, -5272.0620117, 5392.3510742
4: -1856.2840576, 3516.2121582, -1756.1176758, 3330.6079102, -5186.8920898, 5272.3300781

Time for backsubstitution: 1.67 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B1_A2_B2_A2_B1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2776312, upper bound: 4905.2768165
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2778845, upper bound: 4905.2771559
time: 0.81 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1391.7529297, 4864.5195312, -1427.4300537, 4995.9721680, -6387.7250977, 6291.9497070
1: -1394.6345215, 3093.1245117, -1430.5560303, 3173.6352539, -4568.2695312, 4523.6806641
2: -1269.8536377, 3037.1958008, -1303.2268066, 3118.2067871, -4388.0605469, 4340.4228516
3: -1517.6877441, 3681.1262207, -1556.1156006, 3777.5563965, -5295.2421875, 5237.2412109
4: -1761.3310547, 3337.3403320, -1807.7659912, 3424.2595215, -5185.5908203, 5145.1064453

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2759416, upper bound: 4905.2762158
time: 1.35 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2753945, upper bound: 4905.2761117
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1418.6389160, 4958.4526367, -1430.5396729, 5006.5434570, -6425.1821289, 6388.9916992
1: -1423.2806396, 3151.2954102, -1433.7298584, 3180.4970703, -4603.7778320, 4585.0253906
2: -1295.6335449, 3095.7067871, -1306.0885010, 3124.8083496, -4420.4418945, 4401.7949219
3: -1548.4130859, 3751.8264160, -1559.5617676, 3785.8203125, -5334.2333984, 5311.3881836
4: -1795.9704590, 3402.8640137, -1811.6289062, 3431.5529785, -5227.5234375, 5214.4921875

Time for backsubstitution: 1.68 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B1_A2_B1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2759301, upper bound: 4905.2762738
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2753709, upper bound: 4905.2762530
time: 1.00 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1403.0914307, 4904.5454102, -1477.4613037, 5166.5278320, -6569.6191406, 6382.0068359
1: -1406.5969238, 3118.5151367, -1481.1021729, 3283.9785156, -4690.5747070, 4599.6166992
2: -1280.8342285, 3062.0908203, -1349.9902344, 3224.7502441, -4505.5844727, 4412.0810547
3: -1530.4246826, 3711.7854004, -1611.7844238, 3909.6625977, -5440.0864258, 5323.5698242
4: -1776.2138672, 3364.6516113, -1871.0708008, 3542.3005371, -5318.5146484, 5235.7226562

Time for backsubstitution: 1.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2676452, upper bound: 4905.2686669
time: 0.81 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2779380, upper bound: 4905.2781321
time: 0.78 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1423.2420654, 4974.9033203, -1471.0548096, 5144.4399414, -6567.6821289, 6445.9570312
1: -1427.8807373, 3163.3359375, -1475.1716309, 3269.6435547, -4697.5244141, 4638.5063477
2: -1300.2391357, 3105.7895508, -1344.5705566, 3210.6508789, -4510.8901367, 4450.3603516
3: -1553.7158203, 3765.8090820, -1605.3248291, 3892.8520508, -5446.5678711, 5371.1337891
4: -1802.3770752, 3412.7612305, -1863.3255615, 3526.7216797, -5329.0986328, 5276.0854492

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A1_B2_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2678670, upper bound: 4905.2687454
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2787408, upper bound: 4905.2786033
time: 0.87 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1428.6389160, 4990.1406250, -1423.0562744, 4980.9858398, -6409.6245117, 6413.1967773
1: -1431.9676514, 3173.7963867, -1426.2347412, 3163.9985352, -4595.9658203, 4600.0307617
2: -1303.3996582, 3117.2041016, -1299.2072754, 3108.7868652, -4412.1860352, 4416.4111328
3: -1558.2344971, 3777.2927246, -1551.3978271, 3766.1999512, -5324.4335938, 5328.6904297
4: -1806.9791260, 3424.8334961, -1802.1651611, 3413.9472656, -5220.9262695, 5226.9985352

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2630303, upper bound: 4905.2623162
time: 0.83 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A1_B1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2627472, upper bound: 4905.2630528
time: 0.95 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2678987, upper bound: 4905.2660677
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1452.1396484, 5075.5756836, -1426.8474121, 4994.0527344, -6446.1914062, 6502.4228516
1: -1456.6097412, 3227.2126465, -1430.1101074, 3172.3884277, -4628.9980469, 4657.3227539
2: -1327.7854004, 3170.5952148, -1302.7145996, 3116.9160156, -4444.7011719, 4473.3095703
3: -1585.4429932, 3842.2573242, -1555.5936279, 3776.3056641, -5361.7485352, 5397.8500977
4: -1839.8991699, 3483.6367188, -1806.9332275, 3422.9179688, -5262.8159180, 5290.5693359

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B1_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2751328, upper bound: 4905.2744352
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2745148, upper bound: 4905.2744067
time: 0.89 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1443.0363770, 5040.2529297, -1475.4503174, 5159.5283203, -6602.5644531, 6515.7031250
1: -1446.6966553, 3205.7722168, -1479.0064697, 3279.4189453, -4726.1157227, 4684.7783203
2: -1316.4315186, 3148.7656250, -1347.7416992, 3220.4440918, -4536.8754883, 4496.5073242
3: -1574.0736084, 3815.4360352, -1609.5086670, 3904.1066895, -5478.1791992, 5424.9448242
4: -1824.9804688, 3459.5405273, -1868.1123047, 3537.6171875, -5362.5976562, 5327.6523438

Time for backsubstitution: 1.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A1_B1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2721156, upper bound: 4905.2692062
time: 1.32 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2722200, upper bound: 4905.2696891
time: 1.07 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1467.5303955, 5128.7885742, -1479.2164307, 5172.4633789, -6639.9931641, 6608.0048828
1: -1472.2735596, 3261.1418457, -1482.8571777, 3287.7265625, -4760.0000000, 4743.9980469
2: -1341.6121826, 3204.0810547, -1351.2485352, 3228.4682617, -4570.0805664, 4555.3295898
3: -1602.3645020, 3882.6535645, -1613.6934814, 3914.1347656, -5516.4985352, 5496.3471680
4: -1858.9941406, 3520.4626465, -1872.8653564, 3546.5239258, -5405.5175781, 5393.3281250

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A1_B2_A2_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2776051, upper bound: 4905.2767153
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2778845, upper bound: 4905.2775097
time: 1.03 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1374.3929443, 4803.0102539, -1367.4644775, 4779.6000977, -6153.9931641, 6170.4746094
1: -1378.9858398, 3053.8889160, -1371.1257324, 3037.3229980, -4416.3085938, 4425.0146484
2: -1254.2044678, 2999.5278320, -1245.4486084, 2983.9355469, -4238.1401367, 4244.9765625
3: -1498.8746338, 3634.1333008, -1490.1207275, 3613.6718750, -5112.5463867, 5124.2529297
4: -1738.8553467, 3295.4609375, -1728.1635742, 3277.7351074, -5016.5898438, 5023.6240234

Time for backsubstitution: 1.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2694493, upper bound: 4905.2683351
time: 0.78 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2698417, upper bound: 4905.2683960
time: 0.84 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1401.9888916, 4898.7158203, -1370.6166992, 4789.8276367, -6191.8164062, 6269.3325195
1: -1407.6685791, 3114.9809570, -1374.6906738, 3043.4377441, -4451.1054688, 4489.6718750
2: -1280.4139404, 3058.9599609, -1248.3347168, 2989.8325195, -4270.2465820, 4307.2944336
3: -1530.2934570, 3707.4460449, -1494.0416260, 3621.4406738, -5151.7329102, 5201.4877930
4: -1774.5394287, 3361.2104492, -1731.9217529, 3284.4465332, -5058.9858398, 5093.1323242

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B1_A2_B1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2663709, upper bound: 4905.2673247
time: 1.15 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2786651, upper bound: 4905.2790054
time: 0.99 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1388.4277344, 4852.3940430, -1406.5346680, 4915.7661133, -6304.1938477, 6258.9282227
1: -1393.1745605, 3084.8129883, -1411.2412109, 3124.7351074, -4517.9096680, 4496.0537109
2: -1266.8111572, 3030.1760254, -1282.8238525, 3069.5009766, -4336.3120117, 4312.9995117
3: -1514.1524658, 3670.9804688, -1533.8045654, 3718.4218750, -5232.5737305, 5204.7851562
4: -1756.3828125, 3329.1689453, -1778.8748779, 3372.4445801, -5128.8271484, 5108.0439453

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A1_B1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2673910, upper bound: 4905.2683767
time: 0.80 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2785697, upper bound: 4905.2790177
time: 0.95 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1415.1596680, 4945.1381836, -1408.0705566, 4920.8325195, -6335.9912109, 6353.2080078
1: -1420.9591064, 3144.0202637, -1413.1961670, 3127.4675293, -4548.4267578, 4557.2163086
2: -1292.2291260, 3087.7751465, -1284.2888184, 3072.0712891, -4364.3002930, 4372.0639648
3: -1544.6184082, 3742.1665039, -1535.8946533, 3722.1989746, -5266.8173828, 5278.0610352
4: -1791.0076904, 3392.8300781, -1780.7642822, 3375.3740234, -5166.3818359, 5173.5937500

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 47

## Relational analysis of NS_A2_B1_A1_B2_A2_B1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2674994, upper bound: 4905.2684239
time: 0.87 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2789665, upper bound: 4905.2790272
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1408.4080811, 4919.6020508, -1365.3071289, 4771.9096680, -6180.3173828, 6284.9091797
1: -1413.4661865, 3128.0451660, -1369.0097656, 3032.3144531, -4445.7807617, 4497.0546875
2: -1284.6131592, 3073.5981445, -1243.2053223, 2979.2446289, -4263.8574219, 4316.8037109
3: -1536.7302246, 3722.7668457, -1487.8999023, 3607.7905273, -5144.5205078, 5210.6665039
4: -1780.7248535, 3376.5556641, -1725.1134033, 3272.7307129, -5053.4550781, 5101.6689453

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2681505, upper bound: 4905.2674368
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2681680, upper bound: 4905.2675424
time: 0.98 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1451.2263184, 5069.9169922, -1368.5026855, 4782.5883789, -6233.8139648, 6438.4199219
1: -1456.4903564, 3221.9084473, -1372.2651367, 3039.3146973, -4495.8051758, 4594.1728516
2: -1325.0756836, 3166.3085938, -1246.1722412, 2985.9589844, -4311.0341797, 4412.4809570
3: -1583.9223633, 3836.4704590, -1491.5202637, 3616.2172852, -5200.1386719, 5327.9902344
4: -1836.1519775, 3479.2053223, -1729.0766602, 3280.2160645, -5116.3681641, 5208.2817383

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2778087, upper bound: 4905.2776842
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2778090, upper bound: 4905.2782159
time: 0.93 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1422.4885254, 4969.1210938, -1404.3408203, 4908.0800781, -6330.5683594, 6373.4619141
1: -1427.6936035, 3159.0590820, -1409.1052246, 3119.6740723, -4547.3666992, 4568.1630859
2: -1297.1983643, 3104.3937988, -1280.5172119, 3064.7316895, -4361.9296875, 4384.9111328
3: -1552.0462646, 3759.8811035, -1531.5158691, 3712.4938965, -5264.5400391, 5291.3969727
4: -1798.2475586, 3410.3537598, -1775.7980957, 3367.2868652, -5165.5341797, 5186.1513672

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 16

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A1_B1

### Relational analysis result of NS_A2_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2686780, upper bound: 4905.2676658
time: 0.97 seconds

## Relational analysis of NS_A2_B1_A2_B2_A1_B2

### Relational analysis result of NS_A2_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2687298, upper bound: 4905.2677111
time: 0.85 seconds

## BFS NS instance: NS_A2_B1_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1466.0616455, 5121.8183594, -1407.5363770, 4918.6401367, -6384.7016602, 6529.3544922
1: -1471.4354248, 3254.5397949, -1412.3623047, 3126.6477051, -4598.0820312, 4666.9023438
2: -1338.3934326, 3198.5429688, -1283.4710693, 3071.4052734, -4409.7988281, 4482.0141602
3: -1600.0856934, 3875.5136719, -1535.1553955, 3720.8767090, -5320.9624023, 5410.6689453
4: -1854.7036133, 3514.6381836, -1779.7268066, 3374.7414551, -5229.4453125, 5294.3647461

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B1_A2_B2_A2_B1

### Relational analysis result of NS_A2_B1_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2780868, upper bound: 4905.2778973
time: 0.95 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2782516, upper bound: 4905.2782841
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1373.9454346, 4801.0595703, -1442.1597900, 5046.7602539, -6420.7055664, 6243.2192383
1: -1378.4312744, 3052.7338867, -1445.0406494, 3206.3259277, -4584.7573242, 4497.7729492
2: -1253.2868652, 2998.4848633, -1316.4014893, 3149.9587402, -4403.2456055, 4314.8862305
3: -1498.1612549, 3632.4453125, -1571.7985840, 3816.2609863, -5314.4223633, 5204.2431641
4: -1737.7652588, 3294.2985840, -1826.1221924, 3459.0073242, -5196.7724609, 5120.4208984

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2760250, upper bound: 4905.2764851
time: 0.86 seconds

## Relational analysis of NS_A2_B2_A1_B1_A1_B2

### Relational analysis result of NS_A2_B2_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2754613, upper bound: 4905.2761735
time: 1.07 seconds

## BFS NS instance: NS_A2_B2_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1426.3255615, 4983.4057617, -1445.2254639, 5057.0864258, -6483.4121094, 6428.6313477
1: -1431.3885498, 3165.5734863, -1448.1802979, 3213.0656738, -4644.4541016, 4613.7539062
2: -1301.1262207, 3110.4960938, -1319.1965332, 3156.4448242, -4457.5712891, 4429.6918945
3: -1555.9713135, 3769.3820801, -1575.2126465, 3824.3574219, -5380.3276367, 5344.5942383
4: -1803.6246338, 3419.0000000, -1829.8895264, 3466.1872559, -5269.8110352, 5248.8891602

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A1_B1_A2_B1

### Relational analysis result of NS_A2_B2_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2760250, upper bound: 4905.2769497
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2_B2

### Relational analysis result of NS_A2_B2_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2754613, upper bound: 4905.2769481
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1389.1160889, 4854.5048828, -1491.2335205, 5213.6538086, -6602.7695312, 6345.7382812
1: -1393.7833252, 3086.2980957, -1494.4791260, 3314.3020020, -4708.0849609, 4580.7773438
2: -1266.9947510, 3031.7614746, -1362.0408936, 3254.2480469, -4521.2426758, 4393.8012695
3: -1514.7175293, 3672.5090332, -1626.3641357, 3945.4450684, -5460.1621094, 5298.8730469
4: -1756.8350830, 3330.8081055, -1887.9880371, 3574.7475586, -5331.5825195, 5218.7958984

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2786939, upper bound: 4905.2777757
time: 1.04 seconds

## Relational analysis of NS_A2_B2_A1_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2789568, upper bound: 4905.2785534
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1441.5617676, 5036.8476562, -1495.0743408, 5226.6196289, -6668.1806641, 6531.9213867
1: -1446.7832031, 3199.2038574, -1498.4265137, 3322.6694336, -4769.4511719, 4697.6303711
2: -1314.8883057, 3143.6806641, -1365.5473633, 3262.3327637, -4577.2211914, 4509.2280273
3: -1572.5883789, 3809.5747070, -1630.6608887, 3955.5354004, -5528.1240234, 5440.2353516
4: -1822.8070068, 3455.4755859, -1892.7142334, 3583.7561035, -5406.5629883, 5348.1894531

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A1_B2_A2_B1

### Relational analysis result of NS_A2_B2_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2787451, upper bound: 4905.2780917
time: 0.93 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2_B2

### Relational analysis result of NS_A2_B2_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2789665, upper bound: 4905.2789769
time: 0.88 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1403.0382080, 4901.4165039, -1437.5524902, 5030.9697266, -6434.0078125, 6338.9687500
1: -1408.1787109, 3116.6801758, -1440.5156250, 3196.1655273, -4604.3427734, 4557.1958008
2: -1280.0805664, 3062.4240723, -1312.1870117, 3140.0517578, -4420.1323242, 4374.6113281
3: -1530.9034424, 3709.2426758, -1566.8604736, 3804.3081055, -5335.2114258, 5276.1030273
4: -1774.3056641, 3363.9733887, -1820.2338867, 3448.1745605, -5222.4799805, 5184.2065430

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 17
type: B, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 48

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2655568, upper bound: 4905.2643705
time: 0.87 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 7

## Relational analysis of NS_A2_B2_A2_B1_A1_B1

### Relational analysis result of NS_A2_B2_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2611691, upper bound: 4905.2626255
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2_B1_A1_B2

### Relational analysis result of NS_A2_B2_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2659687, upper bound: 4905.2654050
time: 0.84 seconds

## BFS NS instance: NS_A2_B2_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1446.4217529, 5053.9775391, -1441.2908936, 5043.8349609, -6490.2563477, 6495.2685547
1: -1451.7382812, 3211.8244629, -1444.3471680, 3204.4299316, -4656.1679688, 4656.1718750
2: -1321.1489258, 3156.2329102, -1315.6157227, 3148.0742188, -4469.2231445, 4471.8476562
3: -1578.6578369, 3824.6220703, -1571.0134277, 3814.2424316, -5392.8994141, 5395.6357422
4: -1830.6080322, 3467.8403320, -1824.9073486, 3457.0246582, -5287.6328125, 5292.7465820

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B1_A2_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 39

## Relational analysis of NS_A2_B2_A2_B1_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2753484, upper bound: 4905.2763468
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2_B2

### Relational analysis result of NS_A2_B2_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2747524, upper bound: 4905.2763291
time: 0.89 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1419.0141602, 4957.4326172, -1487.2912598, 5200.2734375, -6619.2875977, 6444.7236328
1: -1424.3347168, 3151.9121094, -1490.6256104, 3305.6291504, -4729.9628906, 4642.5375977
2: -1294.4112549, 3097.3242188, -1358.4250488, 3245.8212891, -4540.2324219, 4455.7490234
3: -1548.3133545, 3751.3127441, -1622.1408691, 3935.2707520, -5483.5834961, 5373.4536133
4: -1794.2292480, 3402.2790527, -1882.9532471, 3565.5219727, -5359.7509766, 5285.2324219

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2686773, upper bound: 4905.2675310
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2687298, upper bound: 4905.2676873
time: 0.95 seconds

## BFS NS instance: NS_A2_B2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1462.5131836, 5110.2187500, -1491.7670898, 5215.6347656, -6678.1479492, 6601.9858398
1: -1467.9718018, 3247.2749023, -1495.2296143, 3315.4382324, -4783.4101562, 4742.5029297
2: -1335.6146240, 3191.2282715, -1362.5548096, 3255.3583984, -4590.9731445, 4553.7832031
3: -1596.2011719, 3866.9394531, -1627.1430664, 3947.1154785, -5543.3164062, 5494.0825195
4: -1850.7442627, 3506.2973633, -1888.5531006, 3576.1257324, -5426.8686523, 5394.8505859

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 25
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 44
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 14
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 25

## Relational analysis of NS_A2_B2_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2780348, upper bound: 4905.2774865
time: 1.01 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2782516, upper bound: 4905.2782516
time: 1.56 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 4.50 seconds
NS_A1_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2663945, upper bound: 4905.2666781
NS_A1_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2669861, upper bound: 4905.2669381
NS_A1_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2663045, upper bound: 4905.2671629
NS_A1_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2782680, upper bound: 4905.2779902
NS_A1_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2672286, upper bound: 4905.2681453
NS_A1_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2779380, upper bound: 4905.2778999
NS_A1_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2674484, upper bound: 4905.2682409
NS_A1_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2787408, upper bound: 4905.2783029
NS_A1_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2685644, upper bound: 4905.2675531
NS_A1_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2699893, upper bound: 4905.2680309
NS_A1_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2712149, upper bound: 4905.2700290
NS_A1_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2772618, upper bound: 4905.2763054
NS_A1_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2716380, upper bound: 4905.2691485
NS_A1_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2716431, upper bound: 4905.2692447
NS_A1_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2776312, upper bound: 4905.2768165
NS_A1_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2778845, upper bound: 4905.2771559
NS_A1_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2759416, upper bound: 4905.2762158
NS_A1_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2753945, upper bound: 4905.2761117
NS_A1_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2759301, upper bound: 4905.2762738
NS_A1_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2753709, upper bound: 4905.2762530
NS_A1_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2676452, upper bound: 4905.2686669
NS_A1_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2779380, upper bound: 4905.2781321
NS_A1_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2678670, upper bound: 4905.2687454
NS_A1_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2787408, upper bound: 4905.2786033
NS_A1_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2627472, upper bound: 4905.2630528
NS_A1_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2678987, upper bound: 4905.2660677
NS_A1_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2751328, upper bound: 4905.2744352
NS_A1_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2745148, upper bound: 4905.2744067
NS_A1_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2721156, upper bound: 4905.2692062
NS_A1_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2722200, upper bound: 4905.2696891
NS_A1_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2776051, upper bound: 4905.2767153
NS_A1_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2778845, upper bound: 4905.2775097
NS_A2_B1_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2694493, upper bound: 4905.2683351
NS_A2_B1_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2698417, upper bound: 4905.2683960
NS_A2_B1_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2663709, upper bound: 4905.2673247
NS_A2_B1_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2786651, upper bound: 4905.2790054
NS_A2_B1_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2673910, upper bound: 4905.2683767
NS_A2_B1_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2785697, upper bound: 4905.2790177
NS_A2_B1_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2674994, upper bound: 4905.2684239
NS_A2_B1_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2789665, upper bound: 4905.2790272
NS_A2_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2681505, upper bound: 4905.2674368
NS_A2_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2681680, upper bound: 4905.2675424
NS_A2_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2778087, upper bound: 4905.2776842
NS_A2_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2778090, upper bound: 4905.2782159
NS_A2_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2686780, upper bound: 4905.2676658
NS_A2_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2687298, upper bound: 4905.2677111
NS_A2_B1_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2780868, upper bound: 4905.2778973
NS_A2_B1_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2782516, upper bound: 4905.2782841
NS_A2_B2_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2760250, upper bound: 4905.2764851
NS_A2_B2_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2754613, upper bound: 4905.2761735
NS_A2_B2_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2760250, upper bound: 4905.2769497
NS_A2_B2_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2754613, upper bound: 4905.2769481
NS_A2_B2_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2786939, upper bound: 4905.2777757
NS_A2_B2_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2789568, upper bound: 4905.2785534
NS_A2_B2_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2787451, upper bound: 4905.2780917
NS_A2_B2_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2789665, upper bound: 4905.2789769
NS_A2_B2_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2611691, upper bound: 4905.2626255
NS_A2_B2_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2659687, upper bound: 4905.2654050
NS_A2_B2_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2753484, upper bound: 4905.2763468
NS_A2_B2_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2747524, upper bound: 4905.2763291
NS_A2_B2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2686773, upper bound: 4905.2675310
NS_A2_B2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2687298, upper bound: 4905.2676873
NS_A2_B2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2780348, upper bound: 4905.2774865
NS_A2_B2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 6, time: 4.50
Output dim: 3, lower bound: -4905.2782516, upper bound: 4905.2782516

## BFS NS instance: NS_A1_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1317.9287109, 4599.5654297, -1245.8541260, 4345.3974609, -5663.3256836, 5845.4194336
1: -1321.4729004, 2927.6962891, -1249.6717529, 2763.0954590, -4084.5683594, 4177.3681641
2: -1201.4051514, 2873.2802734, -1132.1204834, 2713.7695312, -3915.1745605, 4005.4006348
3: -1437.7269287, 3483.1345215, -1357.9776611, 3286.5190430, -4724.2451172, 4841.1123047
4: -1665.5863037, 3157.9682617, -1570.2912598, 2982.9580078, -4648.5444336, 4728.2597656

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2663255, upper bound: 4905.2661816
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2663255, upper bound: 4905.2666781
time: 1.02 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1371.8015137, 4792.4223633, -1304.9248047, 4558.7900391, -5930.5908203, 6097.3466797
1: -1374.6845703, 3047.7536621, -1309.3872070, 2898.6582031, -4273.3427734, 4357.1406250
2: -1251.4515381, 2991.8986816, -1189.0616455, 2846.3107910, -4097.7622070, 4180.9599609
3: -1496.2611084, 3627.4104004, -1422.8876953, 3448.9802246, -4945.2412109, 5050.2968750
4: -1735.4880371, 3288.3061523, -1649.4714355, 3126.9243164, -4862.4116211, 4937.7773438

Time for backsubstitution: 1.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2669119, upper bound: 4905.2664659
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2669119, upper bound: 4905.2669381
time: 1.00 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1400.2496338, 4892.8173828, -1328.2282715, 4641.3608398, -6041.6103516, 6221.0458984
1: -1403.8729248, 3111.5817871, -1332.0863037, 2949.1601562, -4353.0327148, 4443.6679688
2: -1278.5095215, 3054.7946777, -1209.2658691, 2897.5800781, -4176.0893555, 4264.0605469
3: -1528.0800781, 3703.5283203, -1447.7987061, 3508.9218750, -5037.0009766, 5151.3261719
4: -1772.5301514, 3356.9794922, -1677.9146729, 3183.1896973, -4955.7197266, 5034.8940430

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2642307, upper bound: 4905.2651914
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2604524, upper bound: 4905.2596655
time: 1.05 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2642236, upper bound: 4905.2652118
time: 0.83 seconds

## BFS NS instance: NS_A1_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1404.4937744, 4907.1840820, -1380.5700684, 4821.7070312, -6226.2006836, 6287.7534180
1: -1408.2156982, 3120.8479004, -1384.8945312, 3061.7285156, -4469.9433594, 4505.7421875
2: -1282.4017334, 3063.7834473, -1257.7041016, 3009.0346680, -4291.4365234, 4321.4873047
3: -1532.8221436, 3714.6418457, -1505.7030029, 3645.5744629, -5178.3964844, 5220.3447266
4: -1777.7945557, 3366.9699707, -1743.8200684, 3307.6894531, -5085.4838867, 5110.7895508

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2734757, upper bound: 4905.2743338
time: 0.83 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2751726, upper bound: 4905.2758467
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2754028, upper bound: 4905.2760980
time: 1.01 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1389.8198242, 4856.6547852, -1365.4334717, 4771.9804688, -6161.8002930, 6222.0874023
1: -1392.7939453, 3088.3698730, -1369.9317627, 3033.3728027, -4426.1669922, 4458.3012695
2: -1267.9766846, 3032.5004883, -1244.9852295, 2980.1552734, -4248.1318359, 4277.4853516
3: -1515.6356201, 3675.5087891, -1488.9156494, 3609.3647461, -5124.9995117, 5164.4238281
4: -1758.6141357, 3332.4653320, -1726.5681152, 3274.3469238, -5032.9609375, 5059.0322266

Time for backsubstitution: 1.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2610296, upper bound: 4905.2614867
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2652033, upper bound: 4905.2661390
time: 0.86 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1393.3981934, 4868.6274414, -1416.1733398, 4947.7851562, -6341.1831055, 6284.7998047
1: -1396.4715576, 3096.1408691, -1421.2021484, 3142.3913574, -4538.8627930, 4517.3427734
2: -1271.2617188, 3039.9965820, -1291.1694336, 3088.4091797, -4359.6708984, 4331.1660156
3: -1519.6491699, 3684.8278809, -1545.0029297, 3741.6142578, -5261.2626953, 5229.8305664
4: -1763.0340576, 3340.8271484, -1790.1467285, 3395.0627441, -5158.0966797, 5130.9731445

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2768447, upper bound: 4905.2771428
time: 0.81 seconds

## Relational analysis of NS_A1_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2761707, upper bound: 4905.2760704
time: 0.92 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1411.6533203, 4933.0224609, -1367.1551514, 4777.8574219, -6189.5102539, 6300.1777344
1: -1415.6860352, 3137.0754395, -1372.0787354, 3036.6035156, -4452.2890625, 4509.1542969
2: -1289.0655518, 3079.9128418, -1246.6445312, 2983.2397461, -4272.3051758, 4326.5576172
3: -1540.7149658, 3734.1010742, -1491.2064209, 3613.7185059, -5154.4331055, 5225.3076172
4: -1787.0914307, 3384.5517578, -1728.7424316, 3277.7722168, -5064.8627930, 5113.2939453

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2668728, upper bound: 4905.2674065
time: 1.00 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2668649, upper bound: 4905.2673958
time: 1.16 seconds

## BFS NS instance: NS_A1_B1_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1415.5504150, 4946.1870117, -1417.1983643, 4950.8920898, -6366.4423828, 6363.3847656
1: -1419.6850586, 3145.5734863, -1422.5659180, 3143.9509277, -4563.6357422, 4568.1391602
2: -1292.6455078, 3088.1557617, -1292.2192383, 3089.6958008, -4382.3413086, 4380.3750000
3: -1545.0850830, 3744.3034668, -1546.3880615, 3743.9616699, -5289.0463867, 5290.6914062
4: -1791.9163818, 3393.7175293, -1791.4071045, 3396.4160156, -5188.3325195, 5185.1230469

Time for backsubstitution: 1.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2772241, upper bound: 4905.2774107
time: 0.91 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2752738, upper bound: 4905.2749094
time: 1.19 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1410.4642334, 4925.7065430, -1313.9022217, 4593.3447266, -6003.8090820, 6239.6083984
1: -1413.4429932, 3132.9614258, -1317.6473389, 2918.9689941, -4332.4106445, 4450.6088867
2: -1286.5595703, 3077.1845703, -1197.0053711, 2868.1435547, -4154.7031250, 4274.1889648
3: -1538.3494873, 3728.5969238, -1432.1022949, 3472.8037109, -5011.1523438, 5160.6982422
4: -1783.6785889, 3381.1823730, -1660.9014893, 3150.4689941, -4934.1474609, 5042.0820312

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2679777, upper bound: 4905.2669385
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2641605, upper bound: 4905.2632372
time: 1.27 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1406.9780273, 4913.5478516, -1346.2720947, 4704.8818359, -6111.8598633, 6259.8198242
1: -1410.3358154, 3125.1982422, -1351.0599365, 2990.0812988, -4400.4169922, 4476.2583008
2: -1283.6629639, 3069.1984863, -1227.0941162, 2937.4208984, -4221.0839844, 4296.2915039
3: -1534.8908691, 3719.5808105, -1468.6577148, 3558.1726074, -5093.0620117, 5188.2382812
4: -1779.5014648, 3372.2448730, -1702.0189209, 3227.1337891, -5006.6352539, 5074.2636719

Time for backsubstitution: 1.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2683141, upper bound: 4905.2671243
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2640901, upper bound: 4905.2632363
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1436.2823486, 5018.8251953, -1316.7907715, 4602.8857422, -6039.1679688, 6335.6162109
1: -1440.5220947, 3191.4975586, -1320.5854492, 2925.2302246, -4365.7514648, 4512.0830078
2: -1313.1077881, 3135.5222168, -1199.6723633, 2874.1064453, -4187.2143555, 4335.1943359
3: -1568.0574951, 3799.6086426, -1435.3645020, 3480.3732910, -5048.4306641, 5234.9731445
4: -1819.5573730, 3445.5136719, -1664.4490967, 3157.1545410, -4976.7119141, 5109.9609375

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2705634, upper bound: 4905.2694813
time: 0.93 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2631045, upper bound: 4905.2619849
time: 1.07 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1432.4029541, 5005.7583008, -1349.1666260, 4714.5097656, -6146.9125977, 6354.9238281
1: -1437.0758057, 3182.7792969, -1354.0383301, 2996.3649902, -4433.4399414, 4536.8173828
2: -1309.9028320, 3126.8930664, -1229.7783203, 2943.4916992, -4253.3935547, 4356.6708984
3: -1564.3145752, 3789.6401367, -1471.9520264, 3565.7351074, -5130.0498047, 5261.5922852
4: -1814.9335938, 3435.8457031, -1705.6022949, 3233.9306641, -5048.8642578, 5141.4482422

Time for backsubstitution: 1.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2719519, upper bound: 4905.2719862
time: 1.01 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2630431, upper bound: 4905.2619560
time: 0.97 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1423.6715088, 4971.8940430, -1356.6568604, 4742.3173828, -6165.9887695, 6328.5502930
1: -1426.9641113, 3162.3747559, -1361.4869385, 3014.5053711, -4441.4697266, 4523.8618164
2: -1298.6289062, 3106.1347656, -1237.6669922, 2961.7104492, -4260.3393555, 4343.8007812
3: -1552.8599854, 3763.7468262, -1479.8104248, 3587.2045898, -5140.0644531, 5243.5571289
4: -1800.3541260, 3412.9733887, -1716.1612549, 3253.9541016, -5054.3081055, 5129.1347656

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2708312, upper bound: 4905.2686621
time: 1.34 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2706077, upper bound: 4905.2682802
time: 1.30 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1420.3836670, 4960.4179688, -1384.9410400, 4840.2319336, -6260.6152344, 6345.3583984
1: -1424.0762939, 3155.0517578, -1390.8134766, 3076.8818359, -4500.9580078, 4545.8652344
2: -1295.8934326, 3098.6230469, -1264.3765869, 3022.3286133, -4318.2207031, 4362.9985352
3: -1549.6546631, 3755.2624512, -1511.9633789, 3662.3234863, -5211.9780273, 5267.2255859
4: -1796.3945312, 3404.5939941, -1752.5594482, 3320.9860840, -5117.3808594, 5157.1518555

Time for backsubstitution: 1.77 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2707981, upper bound: 4905.2686673
time: 1.25 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2696237, upper bound: 4905.2681315
time: 0.88 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1450.0177002, 5066.7358398, -1359.4323730, 4751.4023438, -6201.4199219, 6426.1679688
1: -1454.5567627, 3221.9584961, -1364.3095703, 3020.5017090, -4475.0585938, 4586.2680664
2: -1325.5372314, 3165.6289062, -1240.2205811, 2967.4272461, -4292.9643555, 4405.8496094
3: -1583.2139893, 3835.9721680, -1482.9721680, 3594.4533691, -5177.6669922, 5318.9443359
4: -1836.7291260, 3478.5817871, -1719.5371094, 3260.3750000, -5097.1040039, 5198.1176758

Time for backsubstitution: 1.78 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2767281, upper bound: 4905.2762821
time: 0.92 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2760944, upper bound: 4905.2751322
time: 0.93 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1446.3851318, 5054.4267578, -1388.2841797, 4851.3706055, -6297.7553711, 6442.7109375
1: -1451.3562012, 3213.7260742, -1394.2268066, 3084.1455078, -4535.5009766, 4607.9526367
2: -1322.5517578, 3157.4528809, -1267.4552002, 3029.3496094, -4351.9008789, 4424.9082031
3: -1579.7342529, 3826.5803223, -1515.7539062, 3671.0568848, -5250.7910156, 5342.3339844
4: -1832.3922119, 3469.4194336, -1756.6761475, 3328.8291016, -5161.2211914, 5226.0957031

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2763501, upper bound: 4905.2762721
time: 0.98 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2751011, upper bound: 4905.2748903
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1383.7769775, 4836.6064453, -1412.6054688, 4944.2509766, -6328.0278320, 6249.2114258
1: -1386.6668701, 3075.2844238, -1415.7492676, 3140.5131836, -4527.1801758, 4491.0332031
2: -1262.5487061, 3019.7326660, -1289.6501465, 3085.8039551, -4348.3525391, 4309.3828125
3: -1509.0128174, 3659.9033203, -1539.9625244, 3738.1745605, -5247.1875000, 5199.8657227
4: -1751.2055664, 3318.1838379, -1788.9884033, 3388.6862793, -5139.8916016, 5107.1723633

Time for backsubstitution: 1.79 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2715198, upper bound: 4905.2707333
time: 1.25 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2737094, upper bound: 4905.2739844
time: 0.98 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1379.3798828, 4821.7153320, -1478.2961426, 5168.8222656, -6548.2021484, 6300.0117188
1: -1382.4077148, 3065.4204102, -1483.9406738, 3281.8002930, -4664.2070312, 4549.3608398
2: -1258.4985352, 3010.1708984, -1348.2774658, 3225.0698242, -4483.5683594, 4358.4482422
3: -1504.1582031, 3648.2993164, -1613.3902588, 3907.9316406, -5412.0893555, 5261.6884766
4: -1745.6977539, 3307.6464844, -1869.2325439, 3543.2973633, -5288.9941406, 5176.8789062

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2708915, upper bound: 4905.2707149
time: 1.16 seconds

## Relational analysis of NS_A1_B2_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2721397, upper bound: 4905.2735940
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1410.6743164, 4930.6835938, -1415.7462158, 4954.9414062, -6365.6157227, 6346.4296875
1: -1415.3065186, 3133.5139160, -1418.9532471, 3147.4501953, -4562.7563477, 4552.4667969
2: -1288.3332520, 3078.2988281, -1292.5371094, 3092.4787598, -4380.8120117, 4370.8359375
3: -1539.7291260, 3730.6718750, -1543.4395752, 3746.5212402, -5286.2504883, 5274.1113281
4: -1785.8796387, 3383.7353516, -1792.8929443, 3396.0559082, -5181.9355469, 5176.6284180

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2715197, upper bound: 4905.2705228
time: 0.99 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2736595, upper bound: 4905.2741786
time: 1.06 seconds

## BFS NS instance: NS_A1_B2_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1406.4947510, 4916.4072266, -1481.7307129, 5180.5180664, -6587.0112305, 6398.1376953
1: -1411.2374268, 3124.0986328, -1487.4067383, 3289.3964844, -4700.6318359, 4611.5048828
2: -1284.4714355, 3069.1489258, -1351.4111328, 3232.3527832, -4516.8242188, 4420.5600586
3: -1535.1408691, 3719.5473633, -1617.1414795, 3917.0349121, -5452.1757812, 5336.6889648
4: -1780.5980225, 3373.6784668, -1873.5021973, 3551.3344727, -5331.9326172, 5247.1806641

Time for backsubstitution: 1.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2709102, upper bound: 4905.2705194
time: 0.92 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2721253, upper bound: 4905.2740853
time: 0.95 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1390.3094482, 4859.8808594, -1454.9420166, 5087.7739258, -6478.0834961, 6314.8227539
1: -1393.5954590, 3090.0871582, -1458.2462158, 3233.9240723, -4627.5195312, 4548.3334961
2: -1269.0358887, 3034.2294922, -1329.2307129, 3175.7004395, -4444.7358398, 4363.4599609
3: -1516.3966064, 3677.8652344, -1587.0493164, 3849.8684082, -5366.2651367, 5264.9145508
4: -1759.9769287, 3334.0473633, -1842.4761963, 3488.3977051, -5248.3735352, 5176.5234375

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2633707, upper bound: 4905.2630990
time: 3.36 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2656073, upper bound: 4905.2666483
time: 0.94 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1392.3150635, 4866.3745117, -1484.7957764, 5189.1821289, -6581.4960938, 6351.1704102
1: -1395.6446533, 3094.4243164, -1489.1126709, 3297.9418945, -4693.5864258, 4583.5371094
2: -1270.8933105, 3038.3054199, -1357.2497559, 3239.2163086, -4510.1093750, 4395.5546875
3: -1518.6445312, 3683.0556641, -1620.2740479, 3927.2653809, -5445.9101562, 5303.3286133
4: -1762.4432373, 3338.6252441, -1879.9270020, 3559.5720215, -5322.0146484, 5218.5522461

Time for backsubstitution: 1.81 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2768294, upper bound: 4905.2770524
time: 0.88 seconds

## Relational analysis of NS_A1_B2_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2762329, upper bound: 4905.2766176
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1410.6582031, 4930.8779297, -1448.5692139, 5065.8286133, -6476.4868164, 6379.4462891
1: -1415.0900879, 3135.3513184, -1452.3399658, 3219.6955566, -4634.7851562, 4587.6914062
2: -1288.6694336, 3078.3364258, -1323.8376465, 3161.7053223, -4450.3745117, 4402.1738281
3: -1539.8981934, 3732.4130859, -1580.6138916, 3833.1716309, -5373.0688477, 5313.0268555
4: -1786.4228516, 3382.6096191, -1834.7828369, 3472.9177246, -5259.3398438, 5217.3920898

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2672419, upper bound: 4905.2678768
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2672465, upper bound: 4905.2678857
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A1_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1412.7457275, 4937.7504883, -1476.1986084, 5160.0517578, -6572.7973633, 6413.9492188
1: -1417.2138672, 3139.8847656, -1481.0316162, 3278.9577637, -4696.1718750, 4620.9165039
2: -1290.5950928, 3082.6320801, -1350.0313721, 3220.5876465, -4511.1826172, 4432.6635742
3: -1542.2401123, 3737.8540039, -1611.4010010, 3904.9699707, -5447.2089844, 5349.2548828
4: -1788.9893799, 3387.4147949, -1869.6876221, 3538.7707520, -5327.7583008, 5257.1025391

Time for backsubstitution: 1.82 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2771111, upper bound: 4905.2770868
time: 1.13 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2756949, upper bound: 4905.2751275
time: 0.97 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1402.2644043, 4894.8613281, -1381.7698975, 4833.3676758, -6235.6318359, 6276.6313477
1: -1405.5457764, 3114.0187988, -1384.8679199, 3070.9284668, -4476.4741211, 4498.8857422
2: -1278.5336914, 3058.0612793, -1260.5981445, 3017.1103516, -4295.6440430, 4318.6591797
3: -1529.8543701, 3705.9670410, -1507.2092285, 3655.2731934, -5185.1264648, 5213.1762695
4: -1772.3889160, 3360.8278809, -1748.4724121, 3314.2402344, -5086.6289062, 5109.3002930

Time for backsubstitution: 1.83 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 35

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2587436, upper bound: 4905.2588803
time: 0.82 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2593473, upper bound: 4905.2590316
time: 0.85 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1417.0541992, 4948.6450195, -1413.9703369, 4946.1044922, -6363.1586914, 6362.6152344
1: -1420.4130859, 3147.8229980, -1417.3441162, 3142.5727539, -4562.9858398, 4565.1665039
2: -1292.8325195, 3091.9309082, -1290.4669189, 3087.9924316, -4380.8251953, 4382.3979492
3: -1545.8044434, 3746.1987305, -1541.7264404, 3740.2753906, -5286.0800781, 5287.9252930
4: -1792.0839844, 3397.1228027, -1789.4902344, 3391.6101074, -5183.6943359, 5186.6123047

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 39

### Candidate
type: A, layer: 1, pos: 48

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2607868, upper bound: 4905.2600382
time: 1.09 seconds

## Relational analysis of NS_A1_B2_A2_B1_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2607868, upper bound: 4905.2660677
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1443.8698730, 5046.8076172, -1412.1883545, 4942.9243164, -6386.7939453, 6458.9956055
1: -1448.3317871, 3208.7851562, -1415.4652100, 3139.6435547, -4587.9746094, 4624.2500000
2: -1320.2235107, 3152.5449219, -1289.2825928, 3084.8842773, -4405.1079102, 4441.8271484
3: -1576.4250488, 3820.3183594, -1539.6174316, 3737.3620605, -5313.7861328, 5359.9355469
4: -1829.4372559, 3463.7971191, -1788.3629150, 3387.7485352, -5217.1850586, 5252.1601562

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2707380, upper bound: 4905.2673512
time: 0.97 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2728250, upper bound: 4905.2722070
time: 0.88 seconds

## BFS NS instance: NS_A1_B2_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1441.3402100, 5038.1401367, -1479.8358154, 5174.5185547, -6615.8588867, 6517.9760742
1: -1445.9038086, 3202.9619141, -1485.5632324, 3285.2482910, -4731.1513672, 4688.5244141
2: -1317.8055420, 3146.9294434, -1349.6722412, 3228.3503418, -4546.1547852, 4496.6015625
3: -1573.6484375, 3813.4921875, -1615.1029053, 3912.3339844, -5485.9824219, 5428.5952148
4: -1826.2025146, 3457.6350098, -1871.1035156, 3546.9482422, -5373.1503906, 5328.7382812

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 17

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 25

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2701681, upper bound: 4905.2672840
time: 0.90 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2712885, upper bound: 4905.2721397
time: 1.22 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1426.1699219, 4981.7749023, -1452.6124268, 5080.3208008, -6506.4907227, 6434.3872070
1: -1429.7952881, 3168.4746094, -1456.0666504, 3229.0310059, -4658.8261719, 4624.5410156
2: -1301.4288330, 3112.2136230, -1327.3619385, 3170.8605957, -4472.2895508, 4439.5756836
3: -1555.7797852, 3771.1853027, -1584.6322021, 3844.3261719, -5400.1059570, 5355.8164062
4: -1804.0880127, 3419.3376465, -1839.7789307, 3483.0224609, -5287.1103516, 5259.1166992

Time for backsubstitution: 1.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2712522, upper bound: 4905.2690082
time: 0.86 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2707555, upper bound: 4905.2684000
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1423.7564697, 4973.3588867, -1466.9511719, 5130.0488281, -6553.8051758, 6440.3100586
1: -1427.7944336, 3163.0300293, -1471.8228760, 3260.5393066, -4688.3339844, 4634.8525391
2: -1299.4458008, 3106.5776367, -1341.3996582, 3201.7043457, -4501.1499023, 4447.9775391
3: -1553.5345459, 3764.9875488, -1601.7277832, 3882.3776855, -5435.9121094, 5366.7153320
4: -1801.1799316, 3413.0310059, -1858.4924316, 3517.1225586, -5318.3027344, 5271.5234375

Time for backsubstitution: 1.85 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2712729, upper bound: 4905.2691926
time: 0.91 seconds

## Relational analysis of NS_A1_B2_A2_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2700387, upper bound: 4905.2684476
time: 0.93 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1450.8107910, 5070.7861328, -1456.2871094, 5092.8540039, -6543.6650391, 6527.0732422
1: -1455.5351562, 3224.2668457, -1459.8277588, 3237.1193848, -4692.6542969, 4684.0942383
2: -1326.7275391, 3167.8322754, -1330.7810059, 3178.6572266, -4505.3837891, 4498.6132812
3: -1584.1791992, 3838.8703613, -1588.7183838, 3854.0778809, -5438.2568359, 5427.5883789
4: -1838.3138428, 3480.6425781, -1844.3944092, 3491.6914062, -5330.0053711, 5325.0371094

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2766378, upper bound: 4905.2761228
time: 1.04 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2760264, upper bound: 4905.2751506
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1448.5520020, 5063.3027344, -1471.1047363, 5144.4848633, -6593.0371094, 6534.4072266
1: -1453.7124023, 3219.0593262, -1476.0424805, 3269.7556152, -4723.4672852, 4695.1015625
2: -1324.9635010, 3162.6665039, -1345.2893066, 3210.6376953, -4535.6000977, 4507.9560547
3: -1582.1948242, 3833.0725098, -1606.3105469, 3893.5058594, -5475.7006836, 5439.3828125
4: -1835.6855469, 3474.7844238, -1863.8012695, 3527.0048828, -5362.6884766, 5338.5859375

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 25
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2758028, upper bound: 4905.2756658
time: 1.46 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2760536, upper bound: 4905.2760862
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1314.2738037, 4586.3208008, -1268.3518066, 4424.5473633, -5738.8212891, 5854.6723633
1: -1319.1093750, 2917.0856934, -1272.2404785, 2813.3530273, -4132.4609375, 4189.3261719
2: -1197.1457520, 2864.4355469, -1152.7882080, 2762.9382324, -3960.0839844, 4017.2233887
3: -1433.2530518, 3470.6821289, -1382.3315430, 3346.3479004, -4779.6005859, 4853.0136719
4: -1659.4500732, 3147.9497070, -1599.0058594, 3036.7094727, -4696.1596680, 4746.9555664

Time for backsubstitution: 1.86 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 24

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2694493, upper bound: 4905.2683350
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2694493, upper bound: 4905.2683351
time: 0.89 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1347.2222900, 4707.7905273, -1326.1441650, 4633.8212891, -5981.0434570, 6033.9345703
1: -1352.2847900, 2994.0961914, -1330.5428467, 2946.0363770, -4298.3212891, 4324.6376953
2: -1230.0805664, 2939.7346191, -1208.5491943, 2892.7697754, -4122.8505859, 4148.2836914
3: -1469.8106689, 3563.6108398, -1445.7850342, 3505.5300293, -4975.3398438, 5009.3959961
4: -1705.2227783, 3229.6962891, -1676.6643066, 3177.6831055, -4882.9052734, 4906.3603516

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 47

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 5

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 21

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2672344, upper bound: 4905.2661319
time: 0.85 seconds

## Relational analysis of NS_A2_B1_A1_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2691096, upper bound: 4905.2668077
time: 0.94 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1388.7556152, 4852.3696289, -1347.1231689, 4707.6376953, -6096.3935547, 6199.4926758
1: -1394.2905273, 3085.5388184, -1350.9528809, 2991.1596680, -4385.4497070, 4436.4916992
2: -1268.2370605, 3030.1176758, -1226.5819092, 2938.6552734, -4206.8925781, 4256.6997070
3: -1515.7969971, 3672.3232422, -1468.2421875, 3558.9931641, -5074.7900391, 5140.5649414
4: -1757.7291260, 3329.5595703, -1702.0111084, 3228.2458496, -4985.9750977, 5031.5708008

Time for backsubstitution: 1.87 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2643928, upper bound: 4905.2653448
time: 1.04 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2640924, upper bound: 4905.2643934
time: 0.75 seconds

## BFS NS instance: NS_A2_B1_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1391.6407471, 4861.9658203, -1399.4125977, 4888.0278320, -6279.6684570, 6261.3784180
1: -1397.2440186, 3091.8808594, -1403.7655029, 3103.7722168, -4501.0161133, 4495.6455078
2: -1270.9097900, 3036.2165527, -1275.0283203, 3050.2036133, -4321.1132812, 4311.2451172
3: -1519.0826416, 3679.9184570, -1526.1445312, 3695.6574707, -5214.7397461, 5206.0620117
4: -1761.2932129, 3336.3178711, -1767.8760986, 3352.7971191, -5114.0893555, 5104.1938477

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2773352, upper bound: 4905.2781273
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2672674, upper bound: 4905.2654700
time: 1.40 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1375.4638672, 4807.0834961, -1383.3375244, 4834.6054688, -6210.0693359, 6190.4208984
1: -1380.0589600, 3055.9970703, -1387.7805176, 3073.1591797, -4453.2177734, 4443.7773438
2: -1254.8834229, 3001.9453125, -1261.4553223, 3018.9543457, -4273.8378906, 4263.3999023
3: -1499.9277344, 3636.5827637, -1508.3522949, 3656.8474121, -5156.7749023, 5144.9350586
4: -1739.9400635, 3298.1669922, -1749.4125977, 3316.9653320, -5056.9052734, 5047.5795898

Time for backsubstitution: 1.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 39

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: A, layer: 1, pos: 7

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2626742, upper bound: 4905.2625478
time: 0.84 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2654561, upper bound: 4905.2664240
time: 0.83 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1377.6301270, 4813.9604492, -1433.6062012, 5008.8422852, -6386.4716797, 6247.5659180
1: -1382.2770996, 3060.7187500, -1438.6177979, 3181.2810059, -4563.5581055, 4499.3359375
2: -1256.8969727, 3006.3825684, -1307.2542725, 3126.4101562, -4383.3071289, 4313.6357422
3: -1502.4567871, 3642.2658691, -1563.9296875, 3787.9206543, -5290.3774414, 5206.1948242
4: -1742.5694580, 3303.1618652, -1812.3969727, 3436.7258301, -5179.2949219, 5115.5585938

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2774791, upper bound: 4905.2781574
time: 1.02 seconds

## Relational analysis of NS_A2_B1_A1_B2_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2772148, upper bound: 4905.2778213
time: 1.04 seconds

## BFS NS instance: NS_A2_B1_A1_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1402.0603027, 4899.2661133, -1384.7386475, 4839.2363281, -6241.2963867, 6284.0048828
1: -1407.6995850, 3114.8920898, -1389.5848389, 3075.6147461, -4483.3139648, 4504.4770508
2: -1280.1817627, 3059.2148438, -1262.8059082, 3021.2663574, -4301.4482422, 4322.0200195
3: -1530.2581787, 3707.3962402, -1510.2717285, 3660.2697754, -5190.5268555, 5217.6679688
4: -1774.3795166, 3361.4855957, -1751.1499023, 3319.5883789, -5093.9663086, 5112.6347656

Time for backsubstitution: 1.89 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 44
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 14
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 16
type: A, layer: 1, pos: 17
type: A, layer: 1, pos: 43

Time for candidate selection: 0.15 seconds

### Candidate
type: A, layer: 1, pos: 48

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2668951, upper bound: 4905.2676798
time: 0.99 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2_B1_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -4905.2669273, upper bound: 4905.2676383
time: 1.03 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 4.08 + 416.77 = 420.86 seconds
