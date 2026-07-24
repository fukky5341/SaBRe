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
execution time: IAR + RelationalAnalysis = 1.92 + 13.86 = 15.79 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -82.1305969, upper bound: 82.1305969

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
time: 12.19 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
time: 10.57 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 23.01 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 23.01
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 23.01
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515

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

Time for backsubstitution: 2.18 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 12.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 9.95 seconds

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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 8.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 9.91 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 20.81 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 20.81
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 20.81
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 20.81
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 20.81
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754

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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186311
time: 8.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186309, upper bound: 82.1186332
time: 9.80 seconds

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

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186309
time: 10.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186311, upper bound: 82.1186332
time: 11.50 seconds

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

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186311
time: 10.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186309, upper bound: 82.1186332
time: 12.08 seconds

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

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186309
time: 10.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186311, upper bound: 82.1186332
time: 11.12 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 24.17 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186311
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 1, lower bound: -82.1186309, upper bound: 82.1186332
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186309
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 1, lower bound: -82.1186311, upper bound: 82.1186332
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186311
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 1, lower bound: -82.1186309, upper bound: 82.1186332
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186309
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 24.17
Output dim: 1, lower bound: -82.1186311, upper bound: 82.1186332

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

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962046
time: 8.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962046
time: 8.58 seconds

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

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
time: 9.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
time: 10.72 seconds

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

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962042
time: 9.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962046
time: 9.76 seconds

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

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962015
time: 9.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962011
time: 9.83 seconds

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

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
time: 8.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
time: 9.18 seconds

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

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
time: 10.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
time: 10.67 seconds

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

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
time: 8.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
time: 9.70 seconds

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

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
time: 9.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962011
time: 9.89 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 21.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962046
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962046
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962042
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962046
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962015
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962011
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 21.43
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962011

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

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 10.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.73 seconds

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

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.79 seconds

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

Time for backsubstitution: 1.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 9.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 9.26 seconds

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

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 9.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 9.20 seconds

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

Time for backsubstitution: 1.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.54 seconds

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

Time for backsubstitution: 1.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.60 seconds

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

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 7.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 8.45 seconds

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

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 8.47 seconds

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

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.73 seconds

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

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.93 seconds

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

Time for backsubstitution: 1.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 8.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 8.94 seconds

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

Time for backsubstitution: 1.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 8.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 9.35 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 0
type: DSZ, layer: 1, pos: 120
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 49
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 60
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 78
type: DSZ, layer: 1, pos: 217
type: DSZ, layer: 1, pos: 67
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 117
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 208
type: DSZ, layer: 1, pos: 230
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 218
type: DSZ, layer: 1, pos: 10
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 194
type: DSZ, layer: 1, pos: 4
type: DSZ, layer: 1, pos: 92
type: DSZ, layer: 1, pos: 238
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 136
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 118
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.83 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 18.80 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 18.80
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 18.80
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962011

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 15.79 + 587.09 = 602.88 seconds
