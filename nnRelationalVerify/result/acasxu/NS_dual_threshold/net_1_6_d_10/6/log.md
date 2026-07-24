## Execution arguments:
Dataset: Dataset.ACAS
Network: ds/onnx/acasxu_op11/ACASXU_1_6.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: None
Delta epsilon: 0.1
execution index: (10, None, 6)
Time budget: 420 seconds
Split limit: 100
Threshold: 1406.026249396902


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-225.0351715, 567.3841553, -225.0351715, 567.3841553, -792.4193115, 792.4193115)
1: (-555.2101440, 854.5823364, -555.2101440, 854.5823364, -1409.7924805, 1409.7924805)
2: (-364.9949951, 830.7340698, -364.9949951, 830.7340698, -1195.7290039, 1195.7290039)
3: (-596.9047241, 986.7219238, -596.9047241, 986.7219238, -1583.6267090, 1583.6267090)
4: (-532.2448120, 946.0938721, -532.2448120, 946.0938721, -1478.3386230, 1478.3386230)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.74 + 2.34 = 3.07 seconds
status: Status.UNKNOWN
relational distance
Output dim: 3, lower bound: -1406.0403098, upper bound: 1406.0403098

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 16
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 16

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0364145, upper bound: 1406.0383316
time: 1.24 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0364145, upper bound: 1406.0391278
time: 0.94 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 2.25 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 2.25
Output dim: 3, lower bound: -1406.0364145, upper bound: 1406.0383316
NS_A2, status: Status.UNKNOWN, split count: 1, time: 2.25
Output dim: 3, lower bound: -1406.0364145, upper bound: 1406.0391278

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -219.1331940, 551.8031616, -220.3414764, 555.2274780, -774.3605957, 772.1446533
1: -540.5209961, 830.8065796, -543.5918579, 836.1064453, -1376.6274414, 1374.3984375
2: -355.4857483, 807.5563354, -357.3909302, 812.6823730, -1168.1678467, 1164.9471436
3: -581.3575439, 959.3759155, -584.5502319, 965.4530640, -1546.8104248, 1543.9261475
4: -518.4210205, 919.7958374, -521.2038574, 925.6274414, -1444.0479736, 1440.9997559

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0364141, upper bound: 1406.0364141
time: 0.81 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0364141, upper bound: 1406.0383316
time: 1.13 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -221.0971069, 557.5463257, -222.3280487, 560.6195679, -781.7165527, 779.8742676
1: -545.4309692, 839.8533325, -548.4637451, 844.4490967, -1389.8798828, 1388.3171387
2: -358.5658264, 816.3734131, -360.5691833, 820.8588867, -1179.4246826, 1176.9426270
3: -586.3278809, 969.7855225, -589.6248169, 975.0696411, -1561.3974609, 1559.4101562
4: -522.8688354, 929.8663940, -525.8052368, 934.9271851, -1457.7960205, 1455.6716309

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 16
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 16

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0383316, upper bound: 1406.0364145
time: 1.26 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0383316, upper bound: 1406.0391278
time: 0.91 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 2.94 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -1406.0364141, upper bound: 1406.0364141
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -1406.0364141, upper bound: 1406.0383316
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -1406.0383316, upper bound: 1406.0364145
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 2.94
Output dim: 3, lower bound: -1406.0383316, upper bound: 1406.0391278

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -219.1331940, 551.8031616, -219.1331940, 551.8031616, -770.9362183, 770.9362793
1: -540.5209961, 830.8065796, -540.5209961, 830.8065796, -1371.3275146, 1371.3275146
2: -355.4857483, 807.5563354, -355.4857483, 807.5563354, -1163.0419922, 1163.0421143
3: -581.3575439, 959.3759155, -581.3575439, 959.3759155, -1540.7333984, 1540.7333984
4: -518.4210205, 919.7958374, -518.4210205, 919.7958374, -1438.2167969, 1438.2167969

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0363194, upper bound: 1406.0361776
time: 1.16 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0361085
time: 1.17 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -219.1331940, 551.8031616, -221.0971069, 557.5463257, -776.6793823, 772.9002686
1: -540.5209961, 830.8065796, -545.4309692, 839.8533325, -1380.3742676, 1376.2374268
2: -355.4857483, 807.5563354, -358.5658264, 816.3734131, -1171.8591309, 1166.1221924
3: -581.3575439, 959.3759155, -586.3278809, 969.7855225, -1551.1430664, 1545.7038574
4: -518.4210205, 919.7958374, -522.8688354, 929.8663940, -1448.2873535, 1442.6646729

Time for backsubstitution: 0.69 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0361776, upper bound: 1406.0382724
time: 1.08 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0378425
time: 0.79 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -221.0971069, 557.5463257, -219.1331940, 551.8031616, -772.9002686, 776.6793213
1: -545.4309692, 839.8533325, -540.5209961, 830.8065796, -1376.2374268, 1380.3742676
2: -358.5658264, 816.3734131, -355.4857483, 807.5563354, -1166.1220703, 1171.8591309
3: -586.3278809, 969.7855225, -581.3575439, 959.3759155, -1545.7038574, 1551.1430664
4: -522.8688354, 929.8663940, -518.4210205, 919.7958374, -1442.6646729, 1448.2873535

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0382724, upper bound: 1406.0361783
time: 1.20 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0361085
time: 0.78 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -221.0971069, 557.5463257, -221.0971069, 557.5463257, -778.6433716, 778.6433716
1: -545.4309692, 839.8533325, -545.4309692, 839.8533325, -1385.2843018, 1385.2843018
2: -358.5658264, 816.3734131, -358.5658264, 816.3734131, -1174.9392090, 1174.9392090
3: -586.3278809, 969.7855225, -586.3278809, 969.7855225, -1556.1134033, 1556.1134033
4: -522.8688354, 929.8663940, -522.8688354, 929.8663940, -1452.7352295, 1452.7352295

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: B, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0382724, upper bound: 1406.0384049
time: 1.32 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0382675
time: 0.95 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 3.05 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -1406.0363194, upper bound: 1406.0361776
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0361085
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -1406.0361776, upper bound: 1406.0382724
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0378425
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -1406.0382724, upper bound: 1406.0361783
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0361085
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -1406.0382724, upper bound: 1406.0384049
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 3.05
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0382675

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -204.2511292, 516.1624146, -215.5290222, 542.8892822, -747.1403809, 731.6914062
1: -503.4382324, 777.6658325, -531.4324951, 817.6435547, -1321.0817871, 1309.0983887
2: -331.1529236, 755.6177368, -349.5570984, 794.6088867, -1125.7615967, 1105.1748047
3: -541.1491089, 897.8488159, -571.7969971, 944.0959473, -1485.2448730, 1469.6457520
4: -483.7836304, 860.5120239, -510.0206604, 905.0498657, -1388.8334961, 1370.5327148

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0361085
time: 0.84 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0361085
time: 1.35 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -218.3648224, 549.8685303, -218.6182709, 550.5075073, -768.8723145, 768.4868164
1: -538.6383667, 827.9353638, -539.2593994, 828.8850098, -1367.5233154, 1367.1943359
2: -354.2489929, 804.7556152, -354.6565247, 805.6817627, -1159.9305420, 1159.4121094
3: -579.3477173, 956.0695801, -580.0098267, 957.1625977, -1536.5102539, 1536.0793457
4: -516.6182251, 916.6123047, -517.2133179, 917.6641846, -1434.2822266, 1433.8256836

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0361085
time: 0.86 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0361085
time: 0.96 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -215.5290222, 542.8892822, -206.5020142, 522.7230835, -738.2520752, 749.3912354
1: -531.4324951, 817.6435547, -509.0023193, 787.8549194, -1319.2873535, 1326.6457520
2: -349.5570984, 794.6088867, -334.6948242, 765.6225586, -1115.1796875, 1129.3035889
3: -571.7969971, 944.0959473, -546.8943481, 909.5653076, -1481.3623047, 1490.9902344
4: -510.0206604, 905.0498657, -488.9680786, 871.8327637, -1381.8533936, 1394.0179443

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0378425
time: 1.12 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0378425
time: 1.09 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -218.6182709, 550.5075073, -220.3553772, 555.6460571, -774.2642212, 770.8627319
1: -539.2593994, 828.8850098, -543.5911865, 837.0255127, -1376.2845459, 1372.4761963
2: -354.6565247, 805.6817627, -357.3678284, 813.6072388, -1168.2637939, 1163.0493164
3: -580.0098267, 957.1625977, -584.3914795, 966.5325317, -1546.5423584, 1541.5540771
4: -517.2133179, 917.6641846, -521.1368408, 926.7376099, -1443.9508057, 1438.8007812

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 20
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 20

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0378425
time: 1.37 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0378425
time: 1.34 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -206.5020142, 522.7230835, -215.5290222, 542.8892822, -749.3911743, 738.2520752
1: -509.0023193, 787.8549194, -531.4324951, 817.6435547, -1326.6457520, 1319.2873535
2: -334.6948242, 765.6225586, -349.5570984, 794.6088867, -1129.3035889, 1115.1796875
3: -546.8943481, 909.5653076, -571.7969971, 944.0959473, -1490.9902344, 1481.3623047
4: -488.9680786, 871.8327637, -510.0206604, 905.0498657, -1394.0178223, 1381.8533936

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0361085
time: 1.14 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0361085
time: 1.21 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -220.3553772, 555.6460571, -218.6182709, 550.5075073, -770.8627319, 774.2642212
1: -543.5911865, 837.0255127, -539.2593994, 828.8850098, -1372.4761963, 1376.2845459
2: -357.3678284, 813.6072388, -354.6565247, 805.6817627, -1163.0493164, 1168.2637939
3: -584.3914795, 966.5325317, -580.0098267, 957.1625977, -1541.5540771, 1546.5423584
4: -521.1368408, 926.7376099, -517.2133179, 917.6641846, -1438.8007812, 1443.9508057

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0361085
time: 1.24 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0361085
time: 1.27 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -206.5020142, 522.7230835, -217.5018158, 548.6289673, -755.1307983, 740.2248535
1: -509.0023193, 787.8549194, -536.3890381, 826.6774902, -1335.6798096, 1324.2438965
2: -334.6948242, 765.6225586, -352.6528320, 803.4030151, -1138.0979004, 1118.2753906
3: -546.8943481, 909.5653076, -576.7852783, 954.5651855, -1501.4594727, 1486.3504639
4: -488.9680786, 871.8327637, -514.4527588, 915.1572266, -1404.1252441, 1386.2855225

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0384644, upper bound: 1406.0382675
time: 0.91 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0384644, upper bound: 1406.0382675
time: 1.10 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -220.3553772, 555.6460571, -220.5958099, 556.2631226, -776.6182861, 776.2416992
1: -543.5911865, 837.0255127, -544.1887207, 837.9432983, -1381.5344238, 1381.2142334
2: -357.3678284, 813.6072388, -357.7558899, 814.5073853, -1171.8750000, 1171.3631592
3: -584.3914795, 966.5325317, -585.0183716, 967.5866699, -1551.9781494, 1551.5509033
4: -521.1368408, 926.7376099, -521.6979980, 927.7537231, -1448.8906250, 1448.4355469

Time for backsubstitution: 0.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 20
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 20

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0384644, upper bound: 1406.0382675
time: 1.27 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0384644, upper bound: 1406.0382675
time: 1.01 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 3.06 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0361085
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0361085
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0361085
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0361085
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0378425
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0378425
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0378425
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0361085, upper bound: 1406.0378425
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0361085
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0361085
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0361085
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0378425, upper bound: 1406.0361085
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0384644, upper bound: 1406.0382675
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0384644, upper bound: 1406.0382675
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0384644, upper bound: 1406.0382675
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 3.06
Output dim: 3, lower bound: -1406.0384644, upper bound: 1406.0382675

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -204.2511292, 516.1624146, -204.2511292, 516.1624146, -720.4135742, 720.4135742
1: -503.4382324, 777.6658325, -503.4382324, 777.6658325, -1281.1040039, 1281.1040039
2: -331.1529236, 755.6177368, -331.1529236, 755.6177368, -1086.7706299, 1086.7706299
3: -541.1491089, 897.8488159, -541.1491089, 897.8488159, -1438.9979248, 1438.9979248
4: -483.7836304, 860.5120239, -483.7836304, 860.5120239, -1344.2956543, 1344.2956543

Time for backsubstitution: 0.70 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_B1

### Relational analysis result of NS_A1_B1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0257303, upper bound: 1406.0266371
time: 1.28 seconds

## Relational analysis of NS_A1_B1_A1_B1_B2

### Relational analysis result of NS_A1_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0258845, upper bound: 1406.0255942
time: 1.08 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -204.2511292, 516.1624146, -218.3648224, 549.8685303, -754.1196289, 734.5272217
1: -503.4382324, 777.6658325, -538.6383667, 827.9353638, -1331.3734131, 1316.3041992
2: -331.1529236, 755.6177368, -354.2489929, 804.7556152, -1135.9084473, 1109.8666992
3: -541.1491089, 897.8488159, -579.3477173, 956.0695801, -1497.2186279, 1477.1965332
4: -483.7836304, 860.5120239, -516.6182251, 916.6123047, -1400.3959961, 1377.1302490

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_B1

### Relational analysis result of NS_A1_B1_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0257303, upper bound: 1406.0266371
time: 1.29 seconds

## Relational analysis of NS_A1_B1_A1_B2_B2

### Relational analysis result of NS_A1_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0258845, upper bound: 1406.0255942
time: 1.18 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -218.3648224, 549.8685303, -204.2511292, 516.1624146, -734.5272217, 754.1196289
1: -538.6383667, 827.9353638, -503.4382324, 777.6658325, -1316.3041992, 1331.3734131
2: -354.2489929, 804.7556152, -331.1529236, 755.6177368, -1109.8666992, 1135.9084473
3: -579.3477173, 956.0695801, -541.1491089, 897.8488159, -1477.1965332, 1497.2186279
4: -516.6182251, 916.6123047, -483.7836304, 860.5120239, -1377.1302490, 1400.3959961

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0266366, upper bound: 1406.0254402
time: 0.94 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0255807, upper bound: 1406.0255807
time: 1.21 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -218.3648224, 549.8685303, -218.3648224, 549.8685303, -768.2333374, 768.2333374
1: -538.6383667, 827.9353638, -538.6383667, 827.9353638, -1366.5736084, 1366.5737305
2: -354.2489929, 804.7556152, -354.2489929, 804.7556152, -1159.0045166, 1159.0045166
3: -579.3477173, 956.0695801, -579.3477173, 956.0695801, -1535.4172363, 1535.4172363
4: -516.6182251, 916.6123047, -516.6182251, 916.6123047, -1433.2304688, 1433.2304688

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0266366, upper bound: 1406.0254402
time: 0.99 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0255807, upper bound: 1406.0255807
time: 0.91 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -204.2511292, 516.1624146, -206.5020142, 522.7230835, -726.9742432, 722.6643677
1: -503.4382324, 777.6658325, -509.0023193, 787.8549194, -1291.2932129, 1286.6682129
2: -331.1529236, 755.6177368, -334.6948242, 765.6225586, -1096.7755127, 1090.3125000
3: -541.1491089, 897.8488159, -546.8943481, 909.5653076, -1450.7143555, 1444.7431641
4: -483.7836304, 860.5120239, -488.9680786, 871.8327637, -1355.6164551, 1349.4801025

Time for backsubstitution: 0.72 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_A1_A1

### Relational analysis result of NS_A1_B2_B1_A1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0181758, upper bound: 1406.0152486
time: 1.30 seconds

## Relational analysis of NS_A1_B2_B1_A1_A2

### Relational analysis result of NS_A1_B2_B1_A1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0179545, upper bound: 1406.0151810
time: 1.12 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -218.3648224, 549.8685303, -206.5020142, 522.7230835, -741.0878906, 756.3705444
1: -538.6383667, 827.9353638, -509.0023193, 787.8549194, -1326.4932861, 1336.9375000
2: -354.2489929, 804.7556152, -334.6948242, 765.6225586, -1119.8715820, 1139.4504395
3: -579.3477173, 956.0695801, -546.8943481, 909.5653076, -1488.9130859, 1502.9638672
4: -516.6182251, 916.6123047, -488.9680786, 871.8327637, -1388.4509277, 1405.5803223

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B1_A2_A1

### Relational analysis result of NS_A1_B2_B1_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0181758, upper bound: 1406.0152486
time: 1.37 seconds

## Relational analysis of NS_A1_B2_B1_A2_A2

### Relational analysis result of NS_A1_B2_B1_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0179545, upper bound: 1406.0151810
time: 1.10 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -204.2511292, 516.1624146, -220.3553772, 555.6460571, -759.8972168, 736.5176392
1: -503.4382324, 777.6658325, -543.5911865, 837.0255127, -1340.4637451, 1321.2570801
2: -331.1529236, 755.6177368, -357.3678284, 813.6072388, -1144.7601318, 1112.9855957
3: -541.1491089, 897.8488159, -584.3914795, 966.5325317, -1507.6816406, 1482.2402344
4: -483.7836304, 860.5120239, -521.1368408, 926.7376099, -1410.5211182, 1381.6489258

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_B1

### Relational analysis result of NS_A1_B2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0242945, upper bound: 1406.0263574
time: 1.19 seconds

## Relational analysis of NS_A1_B2_B2_A1_B2

### Relational analysis result of NS_A1_B2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0179545, upper bound: 1406.0151810
time: 1.27 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -218.3648224, 549.8685303, -220.3553772, 555.6460571, -774.0108643, 770.2238159
1: -538.6383667, 827.9353638, -543.5911865, 837.0255127, -1375.6638184, 1371.5264893
2: -354.2489929, 804.7556152, -357.3678284, 813.6072388, -1167.8562012, 1162.1232910
3: -579.3477173, 956.0695801, -584.3914795, 966.5325317, -1545.8802490, 1540.4610596
4: -516.6182251, 916.6123047, -521.1368408, 926.7376099, -1443.3557129, 1437.7491455

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A2_A1

### Relational analysis result of NS_A1_B2_B2_A2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0181758, upper bound: 1406.0152486
time: 1.23 seconds

## Relational analysis of NS_A1_B2_B2_A2_A2

### Relational analysis result of NS_A1_B2_B2_A2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0179545, upper bound: 1406.0151810
time: 1.29 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -206.5020142, 522.7230835, -204.2511292, 516.1624146, -722.6643677, 726.9742432
1: -509.0023193, 787.8549194, -503.4382324, 777.6658325, -1286.6682129, 1291.2932129
2: -334.6948242, 765.6225586, -331.1529236, 755.6177368, -1090.3125000, 1096.7755127
3: -546.8943481, 909.5653076, -541.1491089, 897.8488159, -1444.7431641, 1450.7143555
4: -488.9680786, 871.8327637, -483.7836304, 860.5120239, -1349.4801025, 1355.6164551

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B1_B1

### Relational analysis result of NS_A2_B1_A1_B1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0152486, upper bound: 1406.0181758
time: 0.88 seconds

## Relational analysis of NS_A2_B1_A1_B1_B2

### Relational analysis result of NS_A2_B1_A1_B1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0151810, upper bound: 1406.0179545
time: 1.05 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -206.5020142, 522.7230835, -218.3648224, 549.8685303, -756.3705444, 741.0878906
1: -509.0023193, 787.8549194, -538.6383667, 827.9353638, -1336.9375000, 1326.4932861
2: -334.6948242, 765.6225586, -354.2489929, 804.7556152, -1139.4503174, 1119.8715820
3: -546.8943481, 909.5653076, -579.3477173, 956.0695801, -1502.9638672, 1488.9130859
4: -488.9680786, 871.8327637, -516.6182251, 916.6123047, -1405.5803223, 1388.4509277

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 3

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A1_B2_B1

### Relational analysis result of NS_A2_B1_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0152486, upper bound: 1406.0181758
time: 0.91 seconds

## Relational analysis of NS_A2_B1_A1_B2_B2

### Relational analysis result of NS_A2_B1_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0151810, upper bound: 1406.0179545
time: 1.20 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -220.3553772, 555.6460571, -204.2511292, 516.1624146, -736.5176392, 759.8971558
1: -543.5911865, 837.0255127, -503.4382324, 777.6658325, -1321.2570801, 1340.4637451
2: -357.3678284, 813.6072388, -331.1529236, 755.6177368, -1112.9855957, 1144.7601318
3: -584.3914795, 966.5325317, -541.1491089, 897.8488159, -1482.2402344, 1507.6816406
4: -521.1368408, 926.7376099, -483.7836304, 860.5120239, -1381.6489258, 1410.5211182

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 3, lower bound: -1406.0263574, upper bound: 1406.0242945
time: 1.25 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0151810, upper bound: 1406.0179545
time: 1.24 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -220.3553772, 555.6460571, -218.3648224, 549.8685303, -770.2238159, 774.0108643
1: -543.5911865, 837.0255127, -538.6383667, 827.9353638, -1371.5264893, 1375.6638184
2: -357.3678284, 813.6072388, -354.2489929, 804.7556152, -1162.1232910, 1167.8560791
3: -584.3914795, 966.5325317, -579.3477173, 956.0695801, -1540.4610596, 1545.8802490
4: -521.1368408, 926.7376099, -516.6182251, 916.6123047, -1437.7491455, 1443.3557129

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B2_B1

### Relational analysis result of NS_A2_B1_A2_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0152486, upper bound: 1406.0181758
time: 1.28 seconds

## Relational analysis of NS_A2_B1_A2_B2_B2

### Relational analysis result of NS_A2_B1_A2_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0151810, upper bound: 1406.0179545
time: 0.86 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -206.5020142, 522.7230835, -206.5020142, 522.7230835, -729.2250366, 729.2250366
1: -509.0023193, 787.8549194, -509.0023193, 787.8549194, -1296.8571777, 1296.8571777
2: -334.6948242, 765.6225586, -334.6948242, 765.6225586, -1100.3173828, 1100.3173828
3: -546.8943481, 909.5653076, -546.8943481, 909.5653076, -1456.4597168, 1456.4597168
4: -488.9680786, 871.8327637, -488.9680786, 871.8327637, -1360.8007812, 1360.8007812

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0192292, upper bound: 1406.0152958
time: 1.05 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228
time: 1.35 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -206.5020142, 522.7230835, -220.3553772, 555.6460571, -762.1479492, 743.0782471
1: -509.0023193, 787.8549194, -543.5911865, 837.0255127, -1346.0278320, 1331.4460449
2: -334.6948242, 765.6225586, -357.3678284, 813.6072388, -1148.3020020, 1122.9903564
3: -546.8943481, 909.5653076, -584.3914795, 966.5325317, -1513.4268799, 1493.9567871
4: -488.9680786, 871.8327637, -521.1368408, 926.7376099, -1415.7055664, 1392.9696045

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A1_B2_B1

### Relational analysis result of NS_A2_B2_A1_B2_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0152486, upper bound: 1406.0181758
time: 1.10 seconds

## Relational analysis of NS_A2_B2_A1_B2_B2

### Relational analysis result of NS_A2_B2_A1_B2_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228
time: 0.99 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -220.3553772, 555.6460571, -206.5020142, 522.7230835, -743.0782471, 762.1479492
1: -543.5911865, 837.0255127, -509.0023193, 787.8549194, -1331.4460449, 1346.0278320
2: -357.3678284, 813.6072388, -334.6948242, 765.6225586, -1122.9903564, 1148.3020020
3: -584.3914795, 966.5325317, -546.8943481, 909.5653076, -1493.9567871, 1513.4268799
4: -521.1368408, 926.7376099, -488.9680786, 871.8327637, -1392.9696045, 1415.7055664

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0181758, upper bound: 1406.0152958
time: 1.48 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228
time: 1.13 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -220.3553772, 555.6460571, -220.3553772, 555.6460571, -776.0012207, 776.0012207
1: -543.5911865, 837.0255127, -543.5911865, 837.0255127, -1380.6166992, 1380.6166992
2: -357.3678284, 813.6072388, -357.3678284, 813.6072388, -1170.9749756, 1170.9749756
3: -584.3914795, 966.5325317, -584.3914795, 966.5325317, -1550.9240723, 1550.9240723
4: -521.1368408, 926.7376099, -521.1368408, 926.7376099, -1447.8742676, 1447.8742676

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 7
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 31

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0181758, upper bound: 1406.0152958
time: 1.03 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228
time: 1.46 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 3.31 seconds
NS_A1_B1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0257303, upper bound: 1406.0266371
NS_A1_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0258845, upper bound: 1406.0255942
NS_A1_B1_A1_B2_B1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0257303, upper bound: 1406.0266371
NS_A1_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0258845, upper bound: 1406.0255942
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0266366, upper bound: 1406.0254402
NS_A1_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0255807, upper bound: 1406.0255807
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0266366, upper bound: 1406.0254402
NS_A1_B1_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0255807, upper bound: 1406.0255807
NS_A1_B2_B1_A1_A1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0181758, upper bound: 1406.0152486
NS_A1_B2_B1_A1_A2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0179545, upper bound: 1406.0151810
NS_A1_B2_B1_A2_A1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0181758, upper bound: 1406.0152486
NS_A1_B2_B1_A2_A2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0179545, upper bound: 1406.0151810
NS_A1_B2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0242945, upper bound: 1406.0263574
NS_A1_B2_B2_A1_B2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0179545, upper bound: 1406.0151810
NS_A1_B2_B2_A2_A1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0181758, upper bound: 1406.0152486
NS_A1_B2_B2_A2_A2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0179545, upper bound: 1406.0151810
NS_A2_B1_A1_B1_B1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0152486, upper bound: 1406.0181758
NS_A2_B1_A1_B1_B2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0151810, upper bound: 1406.0179545
NS_A2_B1_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0152486, upper bound: 1406.0181758
NS_A2_B1_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0151810, upper bound: 1406.0179545
NS_A2_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0263574, upper bound: 1406.0242945
NS_A2_B1_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0151810, upper bound: 1406.0179545
NS_A2_B1_A2_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0152486, upper bound: 1406.0181758
NS_A2_B1_A2_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0151810, upper bound: 1406.0179545
NS_A2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0192292, upper bound: 1406.0152958
NS_A2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228
NS_A2_B2_A1_B2_B1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0152486, upper bound: 1406.0181758
NS_A2_B2_A1_B2_B2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0181758, upper bound: 1406.0152958
NS_A2_B2_A2_B1_A2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228
NS_A2_B2_A2_B2_A1, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0181758, upper bound: 1406.0152958
NS_A2_B2_A2_B2_A2, status: Status.VERIFIED, split count: 5, time: 3.31
Output dim: 3, lower bound: -1406.0145228, upper bound: 1406.0145228

## BFS NS instance: NS_A1_B1_A1_B1_B1

### Backsubstitution after applying NS history:
0: -201.3292847, 508.7528076, -199.9211121, 505.2083435, -706.5375366, 708.6739502
1: -495.9657593, 766.3746338, -492.4403076, 760.9943848, -1256.9602051, 1258.8149414
2: -326.3291931, 744.7234497, -324.0153809, 739.5028076, -1065.8319092, 1068.7387695
3: -533.2206421, 884.9003906, -529.4426270, 878.7270508, -1411.9476318, 1414.3430176
4: -476.7901917, 848.1343994, -473.4121094, 842.2297974, -1319.0200195, 1321.5465088

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0247656, upper bound: 1406.0247656
time: 1.20 seconds

## Relational analysis of NS_A1_B1_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0247656, upper bound: 1406.0247687
time: 1.09 seconds

## BFS NS instance: NS_A1_B1_A1_B2_B1

### Backsubstitution after applying NS history:
0: -201.3292847, 508.7528076, -214.0573730, 538.9232178, -740.2524414, 722.8101807
1: -495.9657593, 766.3746338, -527.7200928, 811.2714844, -1307.2373047, 1294.0946045
2: -326.3291931, 744.7234497, -347.1528625, 788.6364746, -1114.9656982, 1091.8763428
3: -533.2206421, 884.9003906, -567.7144775, 936.9520264, -1470.1726074, 1452.6148682
4: -476.7901917, 848.1343994, -506.3062439, 898.3263550, -1375.1165771, 1354.4405518

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 6
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 26
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 38
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A1_B2_B1_A1

### Relational analysis result of NS_A1_B1_A1_B2_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0247374, upper bound: 1406.0244112
time: 1.10 seconds

## Relational analysis of NS_A1_B1_A1_B2_B1_A2

### Relational analysis result of NS_A1_B1_A1_B2_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0247374, upper bound: 1406.0244195
time: 1.38 seconds

## BFS NS instance: NS_A1_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -214.0573730, 538.9232178, -201.3292847, 508.7528076, -722.8101807, 740.2524414
1: -527.7200928, 811.2714844, -495.9657593, 766.3746338, -1294.0946045, 1307.2373047
2: -347.1528625, 788.6364746, -326.3291931, 744.7234497, -1091.8763428, 1114.9656982
3: -567.7144775, 936.9520264, -533.2206421, 884.9003906, -1452.6148682, 1470.1726074
4: -506.3062439, 898.3263550, -476.7901917, 848.1343994, -1354.4404297, 1375.1164551

Time for backsubstitution: 0.73 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 40
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 2
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 26
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 9
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 21
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 33
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 41
type: A, layer: 1, pos: 29
type: B, layer: 1, pos: 38
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 43
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 6

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B1_A1_B1

### Relational analysis result of NS_A1_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0244112, upper bound: 1406.0247374
time: 0.88 seconds

## Relational analysis of NS_A1_B1_A2_B1_A1_B2

### Relational analysis result of NS_A1_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0244112, upper bound: 1406.0257171
time: 1.25 seconds

## BFS NS instance: NS_A1_B1_A2_B2_A1

### Backsubstitution after applying NS history:
0: -214.0573730, 538.9232178, -215.4508209, 542.4515381, -756.5089111, 754.3740234
1: -527.7200928, 811.2714844, -531.1887207, 816.6318970, -1344.3516846, 1342.4602051
2: -347.1528625, 788.6364746, -349.4425354, 793.8420410, -1140.9948730, 1138.0789795
3: -567.7144775, 936.9520264, -571.4398193, 943.1051636, -1510.8195801, 1508.3918457
4: -506.3062439, 898.3263550, -509.6448975, 904.2141113, -1410.5203857, 1407.9711914

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 8
type: A, layer: 1, pos: 8
type: B, layer: 1, pos: 32
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 32
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 2
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 34
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 40
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 45
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 45
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: B, layer: 1, pos: 29
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 18
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 9
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 21
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 6
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 7
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 33
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: B, layer: 1, pos: 43
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 5
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A1_B1_A2_B2_A1_B1

### Relational analysis result of NS_A1_B1_A2_B2_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0244112, upper bound: 1406.0244112
time: 1.06 seconds

## Relational analysis of NS_A1_B1_A2_B2_A1_B2

### Relational analysis result of NS_A1_B1_A2_B2_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0244112, upper bound: 1406.0254311
time: 1.11 seconds

## BFS NS instance: NS_A1_B2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -201.3292847, 508.7528076, -216.6692505, 546.3402100, -747.6694946, 725.4220581
1: -495.9657593, 766.3746338, -534.3793945, 822.9238892, -1318.8896484, 1300.7540283
2: -326.3291931, 744.7234497, -351.3376465, 799.8594971, -1126.1885986, 1096.0610352
3: -533.2206421, 884.9003906, -574.5263062, 950.3548584, -1483.5754395, 1459.4262695
4: -476.7901917, 848.1343994, -512.3258667, 911.2608032, -1388.0509033, 1360.4602051

Time for backsubstitution: 0.74 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 22
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 46
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 35
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 15
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 15
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 35
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 46
type: A, layer: 1, pos: 39
type: B, layer: 1, pos: 39
type: B, layer: 1, pos: 23
type: A, layer: 1, pos: 19
type: A, layer: 1, pos: 43
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 19
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 48
type: A, layer: 1, pos: 47
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 48
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 3
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 47
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 24
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 36
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 23
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 31
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 31
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 24
type: B, layer: 1, pos: 3
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: A, layer: 1, pos: 22

## Relational analysis of NS_A1_B2_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_B2_A1_B1_A1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0179152, upper bound: 1406.0151631
time: 1.07 seconds

## Relational analysis of NS_A1_B2_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_B2_A1_B1_A2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0179152, upper bound: 1406.0151810
time: 1.12 seconds

## BFS NS instance: NS_A2_B1_A2_B1_A1

### Backsubstitution after applying NS history:
0: -216.6692505, 546.3402100, -201.3292847, 508.7528076, -725.4220581, 747.6694946
1: -534.3793945, 822.9238892, -495.9657593, 766.3746338, -1300.7540283, 1318.8896484
2: -351.3376465, 799.8594971, -326.3291931, 744.7234497, -1096.0610352, 1126.1885986
3: -574.5263062, 950.3548584, -533.2206421, 884.9003906, -1459.4262695, 1483.5754395
4: -512.3258667, 911.2608032, -476.7901917, 848.1343994, -1360.4602051, 1388.0509033

Time for backsubstitution: 0.75 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 22
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 46
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 35
type: A, layer: 1, pos: 32
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 15
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 15
type: B, layer: 1, pos: 40
type: A, layer: 1, pos: 35
type: A, layer: 1, pos: 8
type: A, layer: 1, pos: 2
type: B, layer: 1, pos: 8
type: B, layer: 1, pos: 46
type: B, layer: 1, pos: 39
type: A, layer: 1, pos: 39
type: A, layer: 1, pos: 23
type: B, layer: 1, pos: 19
type: B, layer: 1, pos: 43
type: A, layer: 1, pos: 6
type: B, layer: 1, pos: 29
type: B, layer: 1, pos: 45
type: A, layer: 1, pos: 19
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 34
type: A, layer: 1, pos: 48
type: B, layer: 1, pos: 47
type: B, layer: 1, pos: 2
type: A, layer: 1, pos: 34
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 48
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 3
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 47
type: B, layer: 1, pos: 26
type: B, layer: 1, pos: 9
type: A, layer: 1, pos: 24
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 32
type: B, layer: 1, pos: 18
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 26
type: A, layer: 1, pos: 18
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 7
type: A, layer: 1, pos: 28
type: A, layer: 1, pos: 40
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 21
type: A, layer: 1, pos: 9
type: B, layer: 1, pos: 33
type: A, layer: 1, pos: 45
type: B, layer: 1, pos: 41
type: B, layer: 1, pos: 23
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 38
type: B, layer: 1, pos: 21
type: B, layer: 1, pos: 5
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 31
type: A, layer: 1, pos: 7
type: B, layer: 1, pos: 28
type: A, layer: 1, pos: 29
type: A, layer: 1, pos: 31
type: A, layer: 1, pos: 5
type: B, layer: 1, pos: 24
type: A, layer: 1, pos: 3
type: A, layer: 1, pos: 38
type: A, layer: 1, pos: 41
type: A, layer: 1, pos: 33
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 6
type: A, layer: 1, pos: 43

Time for candidate selection: 0.06 seconds

### Candidate
type: B, layer: 1, pos: 22

## Relational analysis of NS_A2_B1_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A2_B1_A1_B1
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0151631, upper bound: 1406.0179152
time: 1.05 seconds

## Relational analysis of NS_A2_B1_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A2_B1_A1_B2
Status: Status.VERIFIED
Output dim: 3, lower bound: -1406.0151631, upper bound: 1406.0179619
time: 1.19 seconds

## Summary of splitting at layer (split count: 5)
- Time for NS candidates: 3.07 seconds
NS_A1_B1_A1_B1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0247656, upper bound: 1406.0247656
NS_A1_B1_A1_B1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0247656, upper bound: 1406.0247687
NS_A1_B1_A1_B2_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0247374, upper bound: 1406.0244112
NS_A1_B1_A1_B2_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0247374, upper bound: 1406.0244195
NS_A1_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0244112, upper bound: 1406.0247374
NS_A1_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0244112, upper bound: 1406.0257171
NS_A1_B1_A2_B2_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0244112, upper bound: 1406.0244112
NS_A1_B1_A2_B2_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0244112, upper bound: 1406.0254311
NS_A1_B2_B2_A1_B1_A1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0179152, upper bound: 1406.0151631
NS_A1_B2_B2_A1_B1_A2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0179152, upper bound: 1406.0151810
NS_A2_B1_A2_B1_A1_B1, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0151631, upper bound: 1406.0179152
NS_A2_B1_A2_B1_A1_B2, status: Status.VERIFIED, split count: 6, time: 3.07
Output dim: 3, lower bound: -1406.0151631, upper bound: 1406.0179619

## NS Result
status: Status.VERIFIED
execution time: (base) + (ns) = 3.07 + 111.97 = 115.04 seconds
