## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 82.0484663031


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428)
1: (-49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420)
2: (-63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871)
3: (-70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899)
4: (-64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137)
5: (-54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285)
6: (-54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004)
7: (-61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411)
8: (-73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273)
9: (-51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.36 + 14.07 = 15.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -82.1305969, upper bound: 82.1305969

# Indivdual Split (IS) starts

## BFS IS instance: IS

Time for backsubstitution: 0.00 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 102
type: A, layer: 1, pos: 102
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 102

## Relational analysis of IS_B1

### Relational analysis result of IS_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1222144, upper bound: 82.1232145
time: 14.52 seconds

## Relational analysis of IS_B2

### Relational analysis result of IS_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
time: 10.45 seconds

## Summary of splitting at layer (split count: 0)
- Time for IS candidates: 25.12 seconds
IS_B1, status: Status.UNKNOWN, split count: 1, time: 25.12
Output dim: 1, lower bound: -82.1222144, upper bound: 82.1232145
IS_B2, status: Status.UNKNOWN, split count: 1, time: 25.12
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515

## BFS IS instance: IS_B1

### Backsubstitution after applying IS history:
0: -56.7985649, 46.7652855, -54.4776917, 44.9459305, -101.7444916, 101.2429810
1: -49.9042206, 38.4526215, -47.9494553, 36.7934914, -86.6977081, 86.4020767
2: -63.7191658, 41.6070213, -61.1624146, 39.9283066, -103.6474686, 102.7694397
3: -70.5155945, 36.0443954, -67.7796097, 34.6275024, -105.1430969, 103.8240051
4: -64.2876892, 46.5763397, -61.7929535, 44.6531944, -108.9408798, 108.3692856
5: -54.7288971, 44.6650429, -52.5003166, 42.9182549, -97.6471481, 97.1653595
6: -54.0188065, 51.2609940, -51.8865585, 49.1987419, -103.2175293, 103.1475525
7: -61.0627823, 49.1374741, -58.7344475, 47.1674576, -108.2302322, 107.8719177
8: -73.9399872, 45.3020477, -71.1593094, 43.3780746, -117.3180389, 116.4613495
9: -51.7169037, 53.1263313, -49.5910416, 51.0189133, -102.7358170, 102.7173767

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=248, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B1_A1

### Relational analysis result of IS_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
time: 9.53 seconds

## Relational analysis of IS_B1_A2

### Relational analysis result of IS_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
time: 8.75 seconds

## BFS IS instance: IS_B2

### Backsubstitution after applying IS history:
0: -54.0580521, 44.6209831, -54.9860611, 45.5575523, -99.6156006, 99.6070251
1: -47.5895157, 36.4892654, -48.5045242, 36.8258667, -84.4153748, 84.9937897
2: -60.6996155, 39.6216812, -61.8208237, 40.2686005, -100.9682083, 101.4425049
3: -67.2845993, 34.3738518, -68.8740158, 34.9882507, -102.2728500, 103.2478638
4: -61.3340836, 44.2981491, -62.7420158, 44.9408379, -106.2749176, 107.0401611
5: -52.1013107, 42.6010056, -52.9771118, 43.4660339, -95.5673447, 95.5781174
6: -51.4955826, 48.8206100, -52.5727386, 49.6672478, -101.1628189, 101.3933487
7: -58.3096085, 46.8092842, -59.7077065, 47.6256256, -105.9352341, 106.5169907
8: -70.6534195, 43.0221291, -72.3851395, 43.4672890, -114.1207047, 115.4072647
9: -49.1903191, 50.6355972, -49.9149780, 51.6287270, -100.8190460, 100.5505753

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=118, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=243, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 102
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 102

## Relational analysis of IS_B2_A1

### Relational analysis result of IS_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
time: 8.93 seconds

## Relational analysis of IS_B2_A2

### Relational analysis result of IS_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
time: 9.51 seconds

## Summary of splitting at layer (split count: 1)
- Time for IS candidates: 19.81 seconds
IS_B1_A1, status: Status.UNKNOWN, split count: 2, time: 19.81
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
IS_B1_A2, status: Status.UNKNOWN, split count: 2, time: 19.81
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
IS_B2_A1, status: Status.UNKNOWN, split count: 2, time: 19.81
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
IS_B2_A2, status: Status.UNKNOWN, split count: 2, time: 19.81
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515

## BFS IS instance: IS_B1_A1

### Backsubstitution after applying IS history:
0: -54.4776917, 44.9459305, -54.4776917, 44.9459305, -99.4236145, 99.4236145
1: -47.9494553, 36.7934914, -47.9494553, 36.7934914, -84.7429428, 84.7429428
2: -61.1624146, 39.9283066, -61.1624146, 39.9283066, -101.0907211, 101.0907211
3: -67.7796097, 34.6275024, -67.7796097, 34.6275024, -102.4071045, 102.4071045
4: -61.7929535, 44.6531944, -61.7929535, 44.6531944, -106.4461517, 106.4461517
5: -52.5003166, 42.9182549, -52.5003166, 42.9182549, -95.4185715, 95.4185715
6: -51.8865585, 49.1987419, -51.8865585, 49.1987419, -101.0852966, 101.0852966
7: -58.7344475, 47.1674576, -58.7344475, 47.1674576, -105.9019012, 105.9019012
8: -71.1593094, 43.3780746, -71.1593094, 43.3780746, -114.5373840, 114.5373688
9: -49.5910416, 51.0189133, -49.5910416, 51.0189133, -100.6099548, 100.6099548

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=248, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A1

### Relational analysis result of IS_B1_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1093292, upper bound: 82.1093757
time: 15.01 seconds

## Relational analysis of IS_B1_A1_A2

### Relational analysis result of IS_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1031544, upper bound: 82.1046080
time: 10.92 seconds

## BFS IS instance: IS_B1_A2

### Backsubstitution after applying IS history:
0: -54.9860611, 45.5575523, -54.4776917, 44.9459305, -99.9319839, 100.0352478
1: -48.5045242, 36.8258667, -47.9494553, 36.7934914, -85.2980118, 84.7753067
2: -61.8208237, 40.2686005, -61.1624146, 39.9283066, -101.7491226, 101.4310074
3: -68.8740158, 34.9882507, -67.7796097, 34.6275024, -103.5015106, 102.7678604
4: -62.7420158, 44.9408379, -61.7929535, 44.6531944, -107.3952103, 106.7337952
5: -52.9771118, 43.4660339, -52.5003166, 42.9182549, -95.8953705, 95.9663467
6: -52.5727386, 49.6672478, -51.8865585, 49.1987419, -101.7714691, 101.5538025
7: -59.7077065, 47.6256256, -58.7344475, 47.1674576, -106.8751450, 106.3600769
8: -72.3851395, 43.4672890, -71.1593094, 43.3780746, -115.7631989, 114.6265869
9: -49.9149780, 51.6287270, -49.5910416, 51.0189133, -100.9338837, 101.2197647

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=118, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=243, inp2_unstable=248, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210290
time: 14.08 seconds

## Relational analysis of IS_B1_A2_B2

### Relational analysis result of IS_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1200299, upper bound: 82.1210409
time: 14.82 seconds

## BFS IS instance: IS_B2_A1

### Backsubstitution after applying IS history:
0: -54.4741821, 44.9435654, -54.9860611, 45.5575523, -100.0317230, 99.9296265
1: -47.9453316, 36.7910461, -48.5045242, 36.8258667, -84.7711792, 85.2955627
2: -61.1589546, 39.9257050, -61.8208237, 40.2686005, -101.4275513, 101.7465286
3: -67.7762527, 34.6247292, -68.8740158, 34.9882507, -102.7645035, 103.4987335
4: -61.7890511, 44.6506691, -62.7420158, 44.9408379, -106.7298889, 107.3926849
5: -52.4970093, 42.9154510, -52.9771118, 43.4660339, -95.9630356, 95.8925629
6: -51.8839188, 49.1943893, -52.5727386, 49.6672478, -101.5511627, 101.7671127
7: -58.7312241, 47.1639862, -59.7077065, 47.6256256, -106.3568344, 106.8716888
8: -71.1548157, 43.3751602, -72.3851395, 43.4672890, -114.6220932, 115.7602997
9: -49.5883102, 51.0159912, -49.9149780, 51.6287270, -101.2170410, 100.9309692

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=118, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=248, inp2_unstable=243, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A1_A1

### Relational analysis result of IS_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1188913, upper bound: 82.1189247
time: 9.76 seconds

## Relational analysis of IS_B2_A1_A2

### Relational analysis result of IS_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 11.36 seconds

## BFS IS instance: IS_B2_A2

### Backsubstitution after applying IS history:
0: -54.9693527, 45.5488586, -54.9860611, 45.5575523, -100.5269012, 100.5349121
1: -48.4966583, 36.8166656, -48.5045242, 36.8258667, -85.3225098, 85.3211823
2: -61.8069763, 40.2592125, -61.8208237, 40.2686005, -102.0755768, 102.0800323
3: -68.8586578, 34.9808235, -68.8740158, 34.9882507, -103.8469086, 103.8548431
4: -62.7267189, 44.9300461, -62.7420158, 44.9408379, -107.6675568, 107.6720581
5: -52.9638786, 43.4559250, -52.9771118, 43.4660339, -96.4299088, 96.4330368
6: -52.5596809, 49.6576233, -52.5727386, 49.6672478, -102.2269211, 102.2303543
7: -59.6970139, 47.6142502, -59.7077065, 47.6256256, -107.3226318, 107.3219376
8: -72.3716278, 43.4590416, -72.3851395, 43.4672890, -115.8389130, 115.8441772
9: -49.9053993, 51.6160812, -49.9149780, 51.6287270, -101.5341263, 101.5310516

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=118, inp2_unstable=118, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=243, inp2_unstable=243, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B2_A2_B1

### Relational analysis result of IS_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1036752, upper bound: 82.1053516
time: 13.74 seconds

## Relational analysis of IS_B2_A2_B2

### Relational analysis result of IS_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1012062, upper bound: 82.1012062
time: 10.74 seconds

## Summary of splitting at layer (split count: 2)
- Time for IS candidates: 25.87 seconds
IS_B1_A1_A1, status: Status.UNKNOWN, split count: 3, time: 25.87
Output dim: 1, lower bound: -82.1093292, upper bound: 82.1093757
IS_B1_A1_A2, status: Status.UNKNOWN, split count: 3, time: 25.87
Output dim: 1, lower bound: -82.1031544, upper bound: 82.1046080
IS_B1_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.87
Output dim: 1, lower bound: -82.1199935, upper bound: 82.1210290
IS_B1_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.87
Output dim: 1, lower bound: -82.1200299, upper bound: 82.1210409
IS_B2_A1_A1, status: Status.UNKNOWN, split count: 3, time: 25.87
Output dim: 1, lower bound: -82.1188913, upper bound: 82.1189247
IS_B2_A1_A2, status: Status.UNKNOWN, split count: 3, time: 25.87
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
IS_B2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 25.87
Output dim: 1, lower bound: -82.1036752, upper bound: 82.1053516
IS_B2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 25.87
Output dim: 1, lower bound: -82.1012062, upper bound: 82.1012062

## BFS IS instance: IS_B1_A1_A1

### Backsubstitution after applying IS history:
0: -52.3560753, 43.2902412, -54.4776917, 44.9459305, -97.3019867, 97.7679138
1: -46.1771202, 35.2787857, -47.9494553, 36.7934914, -82.9706116, 83.2282333
2: -58.8394241, 38.4060135, -61.1624146, 39.9283066, -98.7677231, 99.5684280
3: -65.3081665, 33.3300476, -67.7796097, 34.6275024, -99.9356537, 101.1096420
4: -59.5584297, 42.9072151, -61.7929535, 44.6531944, -104.2116241, 104.7001648
5: -50.4752960, 41.3513374, -52.5003166, 42.9182549, -93.3935547, 93.8516541
6: -49.9505043, 47.3404160, -51.8865585, 49.1987419, -99.1492310, 99.2269745
7: -56.6316833, 45.3864975, -58.7344475, 47.1674576, -103.7991180, 104.1209412
8: -68.6380997, 41.6381683, -71.1593094, 43.3780746, -112.0161743, 112.7974701
9: -47.6718636, 49.1201134, -49.5910416, 51.0189133, -98.6907806, 98.7111511

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=248, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A1_A1_B1

### Relational analysis result of IS_B1_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1182497, upper bound: 82.1182497
time: 10.71 seconds

## Relational analysis of IS_B1_A1_A1_B2

### Relational analysis result of IS_B1_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1182497, upper bound: 82.1182497
time: 8.78 seconds

## BFS IS instance: IS_B1_A1_A2

### Backsubstitution after applying IS history:
0: -48.0582924, 40.0417404, -51.4424515, 42.5760956, -90.6343842, 91.4841919
1: -42.6615829, 32.0045776, -45.4183197, 34.6288834, -77.2904663, 77.4228897
2: -54.2134666, 35.3008652, -57.8432236, 37.7486877, -91.9621429, 93.1440887
3: -60.5978737, 30.6817436, -64.2434998, 32.7647285, -93.3625870, 94.9252319
4: -55.3233681, 39.3112717, -58.5972786, 42.1575851, -97.4809418, 97.9085541
5: -46.3903465, 38.3051987, -49.6068459, 40.6788139, -87.0691605, 87.9120483
6: -46.1548729, 43.5980873, -49.1150703, 46.5433502, -92.6982040, 92.7131577
7: -52.6777420, 41.7869987, -55.7299805, 44.6199875, -97.2977295, 97.5169678
8: -63.8546753, 37.9295578, -67.5444336, 40.8854561, -104.7401276, 105.4739914
9: -43.7283669, 45.3470955, -46.8447380, 48.3004608, -92.0288239, 92.1918106

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=108, inp2_unstable=108, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=233, inp2_unstable=243, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_A2_B1

### Relational analysis result of IS_B1_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1132125, upper bound: 82.1136131
time: 13.11 seconds

## Relational analysis of IS_B1_A1_A2_B2

### Relational analysis result of IS_B1_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1128064, upper bound: 82.1128064
time: 9.00 seconds

## BFS IS instance: IS_B1_A2_B1

### Backsubstitution after applying IS history:
0: -53.0834579, 44.0613899, -48.4329872, 40.2072258, -93.2906799, 92.4943771
1: -46.9003830, 35.4582634, -42.8717613, 32.4849548, -79.3853302, 78.3300095
2: -59.7276421, 38.8954315, -54.5342445, 35.5874519, -95.3150940, 93.4296722
3: -66.6720810, 33.8298035, -60.7848053, 30.9539413, -97.6260223, 94.6146011
4: -60.7131310, 43.3642159, -55.3499374, 39.6578026, -100.3709335, 98.7141418
5: -51.1582222, 42.0598335, -46.7239456, 38.4505692, -89.6087799, 88.7837830
6: -50.8304977, 47.9822807, -46.3644981, 43.8635902, -94.6940918, 94.3467789
7: -57.8161964, 46.0209579, -52.7160759, 42.0955429, -99.9117432, 98.7370300
8: -70.0950699, 41.8829727, -63.9041672, 38.3690300, -108.4640961, 105.7871246
9: -48.1810341, 49.9146805, -44.0978394, 45.5886230, -93.7696533, 94.0125198

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=117, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=241, inp2_unstable=235, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B1_A1

### Relational analysis result of IS_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
time: 14.53 seconds

## Relational analysis of IS_B1_A2_B1_A2

### Relational analysis result of IS_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1210074
time: 15.63 seconds

## BFS IS instance: IS_B1_A2_B2

### Backsubstitution after applying IS history:
0: -52.2830887, 43.4200401, -49.1792641, 40.8133698, -93.0964584, 92.5993042
1: -46.2218704, 34.8948135, -43.5177383, 32.9680634, -79.1899338, 78.4125443
2: -58.8437042, 38.3222656, -55.3701477, 36.1173515, -94.9610596, 93.6924133
3: -65.7255020, 33.3380737, -61.7418861, 31.4068623, -97.1323624, 95.0799561
4: -59.8429985, 42.7075157, -56.2025986, 40.2632446, -100.1062317, 98.9101105
5: -50.3885536, 41.4592171, -47.4304199, 39.0260735, -89.4146271, 88.8896332
6: -50.0914154, 47.2707825, -47.0796127, 44.5302353, -94.6216507, 94.3503876
7: -57.0062294, 45.3439217, -53.5226555, 42.7265549, -99.7327881, 98.8665543
8: -69.1149216, 41.2280540, -64.8939743, 38.9365883, -108.0514984, 106.1220245
9: -47.4487152, 49.1845627, -44.7548294, 46.2781143, -93.7268143, 93.9393768

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=117, inp2_unstable=112, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=240, inp2_unstable=236, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A2_B2_A1

### Relational analysis result of IS_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
time: 14.46 seconds

## Relational analysis of IS_B1_A2_B2_A2

### Relational analysis result of IS_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1210409
time: 9.96 seconds

## BFS IS instance: IS_B2_A1_A1

### Backsubstitution after applying IS history:
0: -48.4329872, 40.2072258, -53.0834579, 44.0613899, -92.4943771, 93.2906799
1: -42.8717613, 32.4849548, -46.9003830, 35.4582634, -78.3300095, 79.3853302
2: -54.5342445, 35.5874519, -59.7276421, 38.8954315, -93.4296722, 95.3150940
3: -60.7848053, 30.9539413, -66.6720810, 33.8298035, -94.6145935, 97.6260223
4: -55.3499374, 39.6578026, -60.7131310, 43.3642159, -98.7141418, 100.3709335
5: -46.7239456, 38.4505692, -51.1582222, 42.0598335, -88.7837830, 89.6087799
6: -46.3644981, 43.8635902, -50.8304977, 47.9822807, -94.3467789, 94.6940918
7: -52.7160759, 42.0955429, -57.8161964, 46.0209579, -98.7370300, 99.9117432
8: -63.9041672, 38.3690300, -70.0950699, 41.8829727, -105.7871246, 108.4640961
9: -44.0978394, 45.5886230, -48.1810341, 49.9146805, -94.0125198, 93.7696533

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=117, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=235, inp2_unstable=241, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B2_A1_A1_B1

### Relational analysis result of IS_B2_A1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1198188
time: 12.75 seconds

## Relational analysis of IS_B2_A1_A1_B2

### Relational analysis result of IS_B2_A1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1199935
time: 15.91 seconds

## BFS IS instance: IS_B2_A1_A2

### Backsubstitution after applying IS history:
0: -49.1792641, 40.8133698, -52.2830887, 43.4200401, -92.5993042, 93.0964584
1: -43.5177383, 32.9680634, -46.2218704, 34.8948135, -78.4125366, 79.1899338
2: -55.3701477, 36.1173515, -58.8437042, 38.3222656, -93.6924133, 94.9610596
3: -61.7418861, 31.4068623, -65.7255020, 33.3380737, -95.0799561, 97.1323624
4: -56.2025986, 40.2632446, -59.8429985, 42.7075157, -98.9101105, 100.1062317
5: -47.4304199, 39.0260735, -50.3885536, 41.4592171, -88.8896332, 89.4146271
6: -47.0796127, 44.5302353, -50.0914154, 47.2707825, -94.3503876, 94.6216507
7: -53.5226555, 42.7265549, -57.0062294, 45.3439217, -98.8665619, 99.7327881
8: -64.8939743, 38.9365883, -69.1149216, 41.2280540, -106.1220245, 108.0514984
9: -44.7548294, 46.2781143, -47.4487152, 49.1845627, -93.9393768, 93.7268143

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=48, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=112, inp2_unstable=117, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=236, inp2_unstable=240, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 181

## Relational analysis of IS_B2_A1_A2_B1

### Relational analysis result of IS_B2_A1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1198188
time: 15.40 seconds

## Relational analysis of IS_B2_A1_A2_B2

### Relational analysis result of IS_B2_A1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1200299
time: 13.17 seconds

## BFS IS instance: IS_B2_A2_B1

### Backsubstitution after applying IS history:
0: -54.9693527, 45.5488586, -52.9419518, 43.9519653, -98.9212875, 98.4908142
1: -48.4966583, 36.8166656, -46.7862663, 35.3519020, -83.8485489, 83.6029205
2: -61.8069763, 40.2592125, -59.5734253, 38.7943649, -100.6013412, 99.8326416
3: -68.8586578, 34.9808235, -66.4872665, 33.7332535, -102.5919113, 101.4680939
4: -62.7267189, 44.9300461, -60.5839005, 43.2527428, -105.9794617, 105.5139465
5: -52.9638786, 43.4559250, -51.0243378, 41.9524193, -94.9162903, 94.4802628
6: -52.5596809, 49.6576233, -50.7019424, 47.8683853, -100.4280472, 100.3595657
7: -59.6970139, 47.6142502, -57.6821404, 45.8971024, -105.5941162, 105.2963867
8: -72.3716278, 43.4590416, -69.9407654, 41.7813873, -114.1530151, 113.3998108
9: -49.9053993, 51.6160812, -48.0561104, 49.7877235, -99.6931229, 99.6721954

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=118, inp2_unstable=117, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=243, inp2_unstable=241, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 181
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.16 seconds

### Candidate
type: A, layer: 1, pos: 114

## Relational analysis of IS_B2_A2_B1_A1

### Relational analysis result of IS_B2_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1012062, upper bound: 82.1012062
time: 7.41 seconds

## Relational analysis of IS_B2_A2_B1_A2

### Relational analysis result of IS_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1012062, upper bound: 82.1012062
time: 9.23 seconds

## BFS IS instance: IS_B2_A2_B2

### Backsubstitution after applying IS history:
0: -52.0930138, 43.2867355, -48.9161758, 40.9216347, -93.0146484, 92.2029114
1: -46.0786819, 34.7457619, -43.5041542, 32.2630653, -78.3417358, 78.2499161
2: -58.6465797, 38.1840858, -55.2470665, 35.8881302, -94.5347137, 93.4311523
3: -65.4999695, 33.2075920, -62.1315041, 31.2522945, -96.7522659, 95.3390961
4: -59.6875572, 42.5553589, -56.6579704, 39.8803635, -99.5679169, 99.2133255
5: -50.2177925, 41.3271217, -47.2064896, 39.1134796, -89.3312683, 88.5336151
6: -49.9231491, 47.1255722, -47.1634598, 44.3677750, -94.2909241, 94.2890244
7: -56.8457413, 45.1814308, -54.0193291, 42.5206375, -99.3663788, 99.2007599
8: -68.9190750, 41.0807495, -65.4753876, 38.2967453, -107.2158203, 106.5561371
9: -47.2875061, 49.0247765, -44.3634720, 46.2676926, -93.5551987, 93.3882446

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=48, inp2_unstable=47, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=117, inp2_unstable=119, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=239, inp2_unstable=226, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.28 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 114
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 36

Time for candidate selection: 0.14 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B2_A2_B2_A1

### Relational analysis result of IS_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0968887, upper bound: 82.0967849
time: 8.72 seconds

## Relational analysis of IS_B2_A2_B2_A2

### Relational analysis result of IS_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0966099, upper bound: 82.0966096
time: 9.52 seconds

## Summary of splitting at layer (split count: 3)
- Time for IS candidates: 19.67 seconds
IS_B1_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1182497, upper bound: 82.1182497
IS_B1_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1182497, upper bound: 82.1182497
IS_B1_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1132125, upper bound: 82.1136131
IS_B1_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1128064, upper bound: 82.1128064
IS_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
IS_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1210074
IS_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1209419
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1210409
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1198188
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1199935
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1198188
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1200299
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1012062, upper bound: 82.1012062
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.1012062, upper bound: 82.1012062
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.0968887, upper bound: 82.0967849
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 19.67
Output dim: 1, lower bound: -82.0966099, upper bound: 82.0966096

## BFS IS instance: IS_B1_A1_A1_B1

### Backsubstitution after applying IS history:
0: -52.3560753, 43.2902412, -52.3560753, 43.2902412, -95.6462936, 95.6463013
1: -46.1771202, 35.2787857, -46.1771202, 35.2787857, -81.4559021, 81.4559021
2: -58.8394241, 38.4060135, -58.8394241, 38.4060135, -97.2454376, 97.2454376
3: -65.3081665, 33.3300476, -65.3081665, 33.3300476, -98.6382065, 98.6381989
4: -59.5584297, 42.9072151, -59.5584297, 42.9072151, -102.4656372, 102.4656448
5: -50.4752960, 41.3513374, -50.4752960, 41.3513374, -91.8266296, 91.8266296
6: -49.9505043, 47.3404160, -49.9505043, 47.3404160, -97.2909088, 97.2909012
7: -56.6316833, 45.3864975, -56.6316833, 45.3864975, -102.0181656, 102.0181656
8: -68.6380997, 41.6381683, -68.6380997, 41.6381683, -110.2762680, 110.2762680
9: -47.6718636, 49.1201134, -47.6718636, 49.1201134, -96.7919693, 96.7919693

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=244, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 250
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 11
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 54

## Relational analysis of IS_B1_A1_A1_B1_A1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 54

## Relational analysis of IS_B1_A1_A1_B1_B1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: B, layer: 1, pos: 190

## Relational analysis of IS_B1_A1_A1_B1_B1

### Relational analysis result of IS_B1_A1_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1154659, upper bound: 82.1144508
time: 14.23 seconds

## Relational analysis of IS_B1_A1_A1_B1_B2

### Relational analysis result of IS_B1_A1_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1152873, upper bound: 82.1141453
time: 13.65 seconds

## BFS IS instance: IS_B1_A1_A1_B2

### Backsubstitution after applying IS history:
0: -52.3560753, 43.2902412, -48.0582924, 40.0417404, -92.3978119, 91.3485184
1: -46.1771202, 35.2787857, -42.6615829, 32.0045776, -78.1817017, 77.9403687
2: -58.8394241, 38.4060135, -54.2134666, 35.3008652, -94.1402893, 92.6194763
3: -65.3081665, 33.3300476, -60.5978737, 30.6817436, -95.9899063, 93.9279022
4: -59.5584297, 42.9072151, -55.3233681, 39.3112717, -98.8697052, 98.2305756
5: -50.4752960, 41.3513374, -46.3903465, 38.3051987, -88.7804871, 87.7416840
6: -49.9505043, 47.3404160, -46.1548729, 43.5980873, -93.5485916, 93.4952774
7: -56.6316833, 45.3864975, -52.6777420, 41.7869987, -98.4186630, 98.0642395
8: -68.6380997, 41.6381683, -63.8546753, 37.9295578, -106.5676575, 105.4928360
9: -47.6718636, 49.1201134, -43.7283669, 45.3470955, -93.0189590, 92.8484802

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=108, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=244, inp2_unstable=233, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 54
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 52
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 181
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 27
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 67
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 11

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_A1_B2_A1

### Relational analysis result of IS_B1_A1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1162963, upper bound: 82.1147106
time: 24.71 seconds

## Relational analysis of IS_B1_A1_A1_B2_A2

### Relational analysis result of IS_B1_A1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1161331, upper bound: 82.1146361
time: 13.06 seconds

## BFS IS instance: IS_B1_A1_A2_B1

### Backsubstitution after applying IS history:
0: -46.2880859, 38.6473694, -45.6264153, 37.9989128, -84.2870026, 84.2737885
1: -41.1662025, 30.7333012, -40.5098953, 30.4660606, -71.6322632, 71.2431946
2: -52.2690697, 34.0269203, -51.4589500, 33.5652618, -85.8343353, 85.4858704
3: -58.5484619, 29.6022549, -57.5008163, 29.2211380, -87.7695999, 87.1030579
4: -53.4299355, 37.8448868, -52.3813400, 37.3415031, -90.7714386, 90.2262192
5: -44.6974869, 36.9869728, -44.0425797, 36.3628845, -81.0603714, 81.0295563
6: -44.5333633, 42.0289307, -43.7920647, 41.3934517, -85.9267960, 85.8209991
7: -50.9071121, 40.2943878, -49.9258957, 39.7244949, -90.6316071, 90.2202835
8: -61.7158966, 36.4572372, -60.5357361, 36.0558281, -97.7717133, 96.9929733
9: -42.1125565, 43.7499352, -41.5430489, 43.0579338, -85.1704865, 85.2929688

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=108, inp2_unstable=108, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=228, inp2_unstable=229, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 85
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 50
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 167
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 230
type: A, layer: 1, pos: 118
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 37
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 136
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 4
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_A2_B1_A1

### Relational analysis result of IS_B1_A1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1128064, upper bound: 82.1128064
time: 10.71 seconds

## Relational analysis of IS_B1_A1_A2_B1_A2

### Relational analysis result of IS_B1_A1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1128064, upper bound: 82.1128064
time: 11.11 seconds

## BFS IS instance: IS_B1_A1_A2_B2

### Backsubstitution after applying IS history:
0: -45.3790550, 37.9238243, -46.2759666, 38.5262108, -83.9052658, 84.1997910
1: -40.3931351, 30.0875435, -41.0707932, 30.8768883, -71.2700195, 71.1583405
2: -51.2658844, 33.3738022, -52.1865234, 34.0241661, -85.2900543, 85.5603180
3: -57.4810181, 29.0431995, -58.3462677, 29.6141052, -87.0951233, 87.3894577
4: -52.4446983, 37.0957603, -53.1291313, 37.8646889, -90.3093872, 90.2248917
5: -43.8249168, 36.2996445, -44.6550484, 36.8661919, -80.6911087, 80.9546661
6: -43.6966019, 41.2202759, -44.4193344, 41.9727249, -85.6693268, 85.6396103
7: -49.9861450, 39.5235176, -50.6350937, 40.2718239, -90.2579498, 90.1585999
8: -60.6070290, 35.7090721, -61.4092674, 36.5414696, -97.1484756, 97.1183167
9: -41.2784424, 42.9207077, -42.1110725, 43.6609497, -84.9393921, 85.0317841

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=108, inp2_unstable=112, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=226, inp2_unstable=229, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: A, layer: 1, pos: 181
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 54
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: A, layer: 1, pos: 251
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 173
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 167
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 85
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 11
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 166
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 42
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 36
type: B, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 181

## Relational analysis of IS_B1_A1_A2_B2_A1

### Relational analysis result of IS_B1_A1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1128064, upper bound: 82.1128064
time: 28.46 seconds

## Relational analysis of IS_B1_A1_A2_B2_A2

### Relational analysis result of IS_B1_A1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1128064, upper bound: 82.1128064
time: 9.69 seconds

## BFS IS instance: IS_B1_A2_B1_A1

### Backsubstitution after applying IS history:
0: -49.0971451, 40.9261818, -48.4329872, 40.2072258, -89.3043594, 89.3591690
1: -43.5386963, 32.5983849, -42.8717613, 32.4849548, -76.0236359, 75.4701385
2: -55.3518600, 36.0223236, -54.5342445, 35.5874519, -90.9393005, 90.5565567
3: -62.0587654, 31.4003963, -60.7848053, 30.9539413, -93.0126953, 92.1851883
4: -56.4579201, 40.0593567, -55.3499374, 39.6578026, -96.1157150, 95.4092865
5: -47.3515167, 39.1028900, -46.7239456, 38.4505692, -85.8020782, 85.8268356
6: -47.1814308, 44.4539680, -46.3644981, 43.8635902, -91.0450211, 90.8184586
7: -53.8425102, 42.6591682, -52.7160759, 42.0955429, -95.9380493, 95.3752441
8: -65.2805557, 38.5690727, -63.9041672, 38.3690300, -103.6495819, 102.4732361
9: -44.5461807, 46.3238754, -44.0978394, 45.5886230, -90.1347961, 90.4217072

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=117, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=235, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 117
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 161
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 49
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 0
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 213
type: B, layer: 1, pos: 78
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 94
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 67
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 118
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 4
type: A, layer: 1, pos: 92
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 76
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_B1_A1_B1

### Relational analysis result of IS_B1_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1004040, upper bound: 82.1030596
time: 13.52 seconds

## Relational analysis of IS_B1_A2_B1_A1_B2

### Relational analysis result of IS_B1_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0987910, upper bound: 82.1005330
time: 12.79 seconds

## BFS IS instance: IS_B1_A2_B1_A2

### Backsubstitution after applying IS history:
0: -50.1318665, 41.7601280, -48.4329872, 40.2072258, -90.3390732, 90.1931152
1: -44.4311523, 33.2874374, -42.8717613, 32.4849548, -76.9161072, 76.1591949
2: -56.5092926, 36.7654228, -54.5342445, 35.5874519, -92.0967255, 91.2996674
3: -63.3537903, 32.0259361, -60.7848053, 30.9539413, -94.3077240, 92.8107376
4: -57.6271362, 40.9063034, -55.3499374, 39.6578026, -97.2849350, 96.2562408
5: -48.3309784, 39.9019165, -46.7239456, 38.4505692, -86.7815475, 86.6258621
6: -48.1643677, 45.3783760, -46.3644981, 43.8635902, -92.0279541, 91.7428665
7: -54.9535789, 43.5358047, -52.7160759, 42.0955429, -97.0491180, 96.2518692
8: -66.6306534, 39.3732300, -63.9041672, 38.3690300, -104.9996796, 103.2773895
9: -45.4693336, 47.2774239, -44.0978394, 45.5886230, -91.0579529, 91.3752594

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=119, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=232, inp2_unstable=235, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 251
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 60
type: A, layer: 1, pos: 190
type: A, layer: 1, pos: 60
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 0
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 120
type: A, layer: 1, pos: 30
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 49
type: A, layer: 1, pos: 217
type: B, layer: 1, pos: 1
type: B, layer: 1, pos: 161
type: B, layer: 1, pos: 13
type: B, layer: 1, pos: 50
type: A, layer: 1, pos: 78
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 178
type: A, layer: 1, pos: 250
type: B, layer: 1, pos: 117
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 194
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 52
type: B, layer: 1, pos: 12
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 120
type: B, layer: 1, pos: 140
type: A, layer: 1, pos: 13
type: B, layer: 1, pos: 217
type: B, layer: 1, pos: 230
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 204
type: B, layer: 1, pos: 176
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 208
type: A, layer: 1, pos: 194
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 204
type: B, layer: 1, pos: 208
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 218
type: B, layer: 1, pos: 94
type: A, layer: 1, pos: 238
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 10
type: B, layer: 1, pos: 10
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 1
type: A, layer: 1, pos: 140
type: A, layer: 1, pos: 176
type: B, layer: 1, pos: 27
type: A, layer: 1, pos: 12
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 238
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 37
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 42
type: B, layer: 1, pos: 92
type: A, layer: 1, pos: 136
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 166
type: B, layer: 1, pos: 68
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_B1_A2_B1

### Relational analysis result of IS_B1_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1004040, upper bound: 82.1030596
time: 14.23 seconds

## Relational analysis of IS_B1_A2_B1_A2_B2

### Relational analysis result of IS_B1_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0987910, upper bound: 82.1005330
time: 11.54 seconds

## BFS IS instance: IS_B1_A2_B2_A1

### Backsubstitution after applying IS history:
0: -49.0971451, 40.9261818, -49.1792641, 40.8133698, -89.9105148, 90.1054459
1: -43.5386963, 32.5983849, -43.5177383, 32.9680634, -76.5067444, 76.1161194
2: -55.3518600, 36.0223236, -55.3701477, 36.1173515, -91.4692078, 91.3924637
3: -62.0587654, 31.4003963, -61.7418861, 31.4068623, -93.4656219, 93.1422729
4: -56.4579201, 40.0593567, -56.2025986, 40.2632446, -96.7211456, 96.2619553
5: -47.3515167, 39.1028900, -47.4304199, 39.0260735, -86.3775940, 86.5333099
6: -47.1814308, 44.4539680, -47.0796127, 44.5302353, -91.7116623, 91.5335846
7: -53.8425102, 42.6591682, -53.5226555, 42.7265549, -96.5690613, 96.1818161
8: -65.2805557, 38.5690727, -64.8939743, 38.9365883, -104.2171478, 103.4630432
9: -44.5461807, 46.3238754, -44.7548294, 46.2781143, -90.8242874, 91.0786896

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=47, inp2_unstable=47, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=117, inp2_unstable=112, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=231, inp2_unstable=236, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### IS candidates at layer 1
type: B, layer: 1, pos: 114
type: B, layer: 1, pos: 190
type: B, layer: 1, pos: 250
type: B, layer: 1, pos: 251
type: A, layer: 1, pos: 54
type: B, layer: 1, pos: 178
type: B, layer: 1, pos: 54
type: A, layer: 1, pos: 114
type: B, layer: 1, pos: 173
type: B, layer: 1, pos: 0
type: B, layer: 1, pos: 30
type: B, layer: 1, pos: 85
type: A, layer: 1, pos: 251
type: B, layer: 1, pos: 11
type: B, layer: 1, pos: 213
type: A, layer: 1, pos: 117
type: B, layer: 1, pos: 120
type: B, layer: 1, pos: 161
type: A, layer: 1, pos: 190
type: B, layer: 1, pos: 60
type: B, layer: 1, pos: 49
type: B, layer: 1, pos: 1
type: A, layer: 1, pos: 60
type: B, layer: 1, pos: 13
type: A, layer: 1, pos: 30
type: A, layer: 1, pos: 78
type: A, layer: 1, pos: 217
type: A, layer: 1, pos: 173
type: B, layer: 1, pos: 50
type: B, layer: 1, pos: 167
type: A, layer: 1, pos: 161
type: B, layer: 1, pos: 12
type: B, layer: 1, pos: 176
type: A, layer: 1, pos: 250
type: A, layer: 1, pos: 178
type: B, layer: 1, pos: 117
type: B, layer: 1, pos: 94
type: B, layer: 1, pos: 140
type: B, layer: 1, pos: 230
type: B, layer: 1, pos: 194
type: A, layer: 1, pos: 49
type: A, layer: 1, pos: 67
type: B, layer: 1, pos: 52
type: A, layer: 1, pos: 13
type: A, layer: 1, pos: 85
type: A, layer: 1, pos: 52
type: A, layer: 1, pos: 0
type: A, layer: 1, pos: 120
type: A, layer: 1, pos: 208
type: B, layer: 1, pos: 218
type: A, layer: 1, pos: 218
type: B, layer: 1, pos: 217
type: A, layer: 1, pos: 27
type: A, layer: 1, pos: 176
type: A, layer: 1, pos: 238
type: B, layer: 1, pos: 208
type: B, layer: 1, pos: 78
type: A, layer: 1, pos: 213
type: A, layer: 1, pos: 94
type: A, layer: 1, pos: 50
type: A, layer: 1, pos: 194
type: B, layer: 1, pos: 204
type: A, layer: 1, pos: 11
type: A, layer: 1, pos: 167
type: A, layer: 1, pos: 118
type: A, layer: 1, pos: 12
type: A, layer: 1, pos: 204
type: A, layer: 1, pos: 140
type: B, layer: 1, pos: 27
type: B, layer: 1, pos: 238
type: A, layer: 1, pos: 10
type: A, layer: 1, pos: 1
type: B, layer: 1, pos: 10
type: B, layer: 1, pos: 67
type: A, layer: 1, pos: 230
type: B, layer: 1, pos: 166
type: B, layer: 1, pos: 118
type: B, layer: 1, pos: 42
type: A, layer: 1, pos: 42
type: A, layer: 1, pos: 76
type: B, layer: 1, pos: 37
type: B, layer: 1, pos: 76
type: A, layer: 1, pos: 136
type: B, layer: 1, pos: 92
type: B, layer: 1, pos: 136
type: A, layer: 1, pos: 4
type: A, layer: 1, pos: 37
type: A, layer: 1, pos: 92
type: B, layer: 1, pos: 4
type: B, layer: 1, pos: 68
type: A, layer: 1, pos: 166
type: A, layer: 1, pos: 68
type: B, layer: 1, pos: 36
type: A, layer: 1, pos: 36

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 114

## Relational analysis of IS_B1_A2_B2_A1_B1

### Relational analysis result of IS_B1_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1003496, upper bound: 82.1029639
time: 12.64 seconds

## Relational analysis of IS_B1_A2_B2_A1_B2

### Relational analysis result of IS_B1_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0984971, upper bound: 82.1000631
time: 11.32 seconds

## Summary of splitting at layer (split count: 4)
- Time for IS candidates: 25.36 seconds
IS_B1_A1_A1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.1154659, upper bound: 82.1144508
IS_B1_A1_A1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.1152873, upper bound: 82.1141453
IS_B1_A1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.1162963, upper bound: 82.1147106
IS_B1_A1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.1161331, upper bound: 82.1146361
IS_B1_A1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.1128064, upper bound: 82.1128064
IS_B1_A1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.1128064, upper bound: 82.1128064
IS_B1_A1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.1128064, upper bound: 82.1128064
IS_B1_A1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.1128064, upper bound: 82.1128064
IS_B1_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.1004040, upper bound: 82.1030596
IS_B1_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.0987910, upper bound: 82.1005330
IS_B1_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.1004040, upper bound: 82.1030596
IS_B1_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.0987910, upper bound: 82.1005330
IS_B1_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.1003496, upper bound: 82.1029639
IS_B1_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 25.36
Output dim: 1, lower bound: -82.0984971, upper bound: 82.1000631
IS_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 1, lower bound: -82.1198188, upper bound: 82.1210409
IS_B2_A1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1198188
IS_B2_A1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1199935
IS_B2_A1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1198188
IS_B2_A1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 1, lower bound: -82.1209419, upper bound: 82.1200299
IS_B2_A2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 1, lower bound: -82.1012062, upper bound: 82.1012062
IS_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 1, lower bound: -82.1012062, upper bound: 82.1012062
IS_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 1, lower bound: -82.0968887, upper bound: 82.0967849
IS_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 25.36
Output dim: 1, lower bound: -82.0966099, upper bound: 82.0966096

## IS Result
status: Status.UNKNOWN
execution time: (base) + (is) = 15.43 + 591.88 = 607.30 seconds
