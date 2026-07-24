## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 82.0484663031


## IAR start

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
execution time: IAR + RelationalAnalysis = 0.87 + 13.61 = 14.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -82.1305969, upper bound: 82.1305969

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 120

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1302624, upper bound: 82.1302590
time: 11.38 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1302591, upper bound: 82.1302624
time: 12.17 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 23.57 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 23.57
Output dim: 1, lower bound: -82.1302624, upper bound: 82.1302590
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 23.57
Output dim: 1, lower bound: -82.1302591, upper bound: 82.1302624

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1206472, upper bound: 82.1206472
time: 10.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1206472, upper bound: 82.1206472
time: 9.80 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1276088, upper bound: 82.1276099
time: 13.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1276104, upper bound: 82.1276082
time: 10.03 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 24.17 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 24.17
Output dim: 1, lower bound: -82.1206472, upper bound: 82.1206472
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 24.17
Output dim: 1, lower bound: -82.1206472, upper bound: 82.1206472
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 24.17
Output dim: 1, lower bound: -82.1276088, upper bound: 82.1276099
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 24.17
Output dim: 1, lower bound: -82.1276104, upper bound: 82.1276082

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1194707, upper bound: 82.1194665
time: 9.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1194665, upper bound: 82.1194707
time: 9.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 92

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1206465, upper bound: 82.1206472
time: 10.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1206472, upper bound: 82.1206465
time: 8.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 0

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1272582, upper bound: 82.1272620
time: 10.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1272575, upper bound: 82.1272619
time: 10.18 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 68

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1241256, upper bound: 82.1241267
time: 9.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1241256, upper bound: 82.1241267
time: 10.70 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 20.92 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.92
Output dim: 1, lower bound: -82.1194707, upper bound: 82.1194665
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.92
Output dim: 1, lower bound: -82.1194665, upper bound: 82.1194707
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.92
Output dim: 1, lower bound: -82.1206465, upper bound: 82.1206472
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.92
Output dim: 1, lower bound: -82.1206472, upper bound: 82.1206465
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.92
Output dim: 1, lower bound: -82.1272582, upper bound: 82.1272620
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.92
Output dim: 1, lower bound: -82.1272575, upper bound: 82.1272619
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 20.92
Output dim: 1, lower bound: -82.1241256, upper bound: 82.1241267
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 20.92
Output dim: 1, lower bound: -82.1241256, upper bound: 82.1241267

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1194697, upper bound: 82.1194665
time: 9.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1194707, upper bound: 82.1194664
time: 18.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1181226, upper bound: 82.1181295
time: 11.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1181283, upper bound: 82.1181242
time: 9.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1175079, upper bound: 82.1175079
time: 10.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1175079, upper bound: 82.1175079
time: 9.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1183837, upper bound: 82.1183803
time: 10.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1183842, upper bound: 82.1183802
time: 11.34 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1245181, upper bound: 82.1245198
time: 12.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1245181, upper bound: 82.1245198
time: 12.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 208

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0915431, upper bound: 82.0915415
time: 8.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0915431, upper bound: 82.0915415
time: 8.49 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0907711, upper bound: 82.0907769
time: 9.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0907712, upper bound: 82.0907769
time: 10.03 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1132620, upper bound: 82.1132604
time: 9.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1132620, upper bound: 82.1132604
time: 11.23 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 21.68 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1194697, upper bound: 82.1194665
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1194707, upper bound: 82.1194664
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1181226, upper bound: 82.1181295
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1181283, upper bound: 82.1181242
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1175079, upper bound: 82.1175079
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1175079, upper bound: 82.1175079
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1183837, upper bound: 82.1183803
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1183842, upper bound: 82.1183802
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1245181, upper bound: 82.1245198
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1245181, upper bound: 82.1245198
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.0915431, upper bound: 82.0915415
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.0915431, upper bound: 82.0915415
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.0907711, upper bound: 82.0907769
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.0907712, upper bound: 82.0907769
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1132620, upper bound: 82.1132604
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.68
Output dim: 1, lower bound: -82.1132620, upper bound: 82.1132604

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 49

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1194697, upper bound: 82.1194665
time: 10.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1194697, upper bound: 82.1194663
time: 10.99 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1140354, upper bound: 82.1140331
time: 9.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1140353, upper bound: 82.1140328
time: 10.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 67

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1181226, upper bound: 82.1181214
time: 11.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1181175, upper bound: 82.1181295
time: 11.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889804, upper bound: 82.0889785
time: 8.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889804, upper bound: 82.0889785
time: 6.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1124696, upper bound: 82.1124713
time: 9.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1124712, upper bound: 82.1124701
time: 11.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0951245, upper bound: 82.0951256
time: 10.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0951245, upper bound: 82.0951256
time: 8.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 0

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1165487, upper bound: 82.1165435
time: 10.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1165463, upper bound: 82.1165456
time: 10.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1166951, upper bound: 82.1166961
time: 11.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1166971, upper bound: 82.1166945
time: 10.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1232844, upper bound: 82.1232799
time: 10.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1232801, upper bound: 82.1232833
time: 11.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1242805, upper bound: 82.1242780
time: 11.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1242788, upper bound: 82.1242799
time: 9.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 136

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0801505, upper bound: 82.0801511
time: 12.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0801505, upper bound: 82.0801511
time: 11.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -56.7985649, 46.7652855, -56.7985649, 46.7652855, -103.5638504, 103.5638428
1: -49.9042206, 38.4526215, -49.9042206, 38.4526215, -88.3568420, 88.3568420
2: -63.7191658, 41.6070213, -63.7191658, 41.6070213, -105.3261871, 105.3261871
3: -70.5155945, 36.0443954, -70.5155945, 36.0443954, -106.5599899, 106.5599899
4: -64.2876892, 46.5763397, -64.2876892, 46.5763397, -110.8640137, 110.8640137
5: -54.7288971, 44.6650429, -54.7288971, 44.6650429, -99.3939285, 99.3939285
6: -54.0188065, 51.2609940, -54.0188065, 51.2609940, -105.2798004, 105.2798004
7: -61.0627823, 49.1374741, -61.0627823, 49.1374741, -110.2002487, 110.2002411
8: -73.9399872, 45.3020477, -73.9399872, 45.3020477, -119.2420273, 119.2420273
9: -51.7169037, 53.1263313, -51.7169037, 53.1263313, -104.8432312, 104.8432312

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 49

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0849951, upper bound: 82.0849940
time: 8.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0849951, upper bound: 82.0849940
time: 9.89 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 19.12 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1194697, upper bound: 82.1194665
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1194697, upper bound: 82.1194663
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1140354, upper bound: 82.1140331
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1140353, upper bound: 82.1140328
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1181226, upper bound: 82.1181214
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1181175, upper bound: 82.1181295
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.0889804, upper bound: 82.0889785
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.0889804, upper bound: 82.0889785
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1124696, upper bound: 82.1124713
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1124712, upper bound: 82.1124701
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.0951245, upper bound: 82.0951256
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.0951245, upper bound: 82.0951256
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1165487, upper bound: 82.1165435
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1165463, upper bound: 82.1165456
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1166951, upper bound: 82.1166961
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1166971, upper bound: 82.1166945
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1232844, upper bound: 82.1232799
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1232801, upper bound: 82.1232833
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1242805, upper bound: 82.1242780
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.1242788, upper bound: 82.1242799
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.0801505, upper bound: 82.0801511
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.0801505, upper bound: 82.0801511
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.0849951, upper bound: 82.0849940
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 19.12
Output dim: 1, lower bound: -82.0849951, upper bound: 82.0849940
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.12
Output dim: 1, lower bound: -82.0907711, upper bound: 82.0907769
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.12
Output dim: 1, lower bound: -82.0907712, upper bound: 82.0907769
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 19.12
Output dim: 1, lower bound: -82.1132620, upper bound: 82.1132604
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 19.12
Output dim: 1, lower bound: -82.1132620, upper bound: 82.1132604

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 14.48 + 585.74 = 600.22 seconds
