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
execution time: IAR + RelationalAnalysis = 1.34 + 13.96 = 15.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -82.1305969, upper bound: 82.1305969

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
time: 12.19 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
time: 10.50 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 22.83 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 22.83
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 22.83
Output dim: 1, lower bound: -82.1212515, upper bound: 82.1212515

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 12.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 9.98 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 9.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
time: 10.15 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 20.52 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.52
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.52
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 20.52
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 20.52
Output dim: 1, lower bound: -82.1190754, upper bound: 82.1190754

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186311
time: 8.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186309, upper bound: 82.1186332
time: 9.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186309
time: 10.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186311, upper bound: 82.1186332
time: 11.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186311
time: 10.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186309, upper bound: 82.1186332
time: 12.22 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186309
time: 11.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1186311, upper bound: 82.1186332
time: 11.13 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 23.52 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186311
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 1, lower bound: -82.1186309, upper bound: 82.1186332
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186309
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 1, lower bound: -82.1186311, upper bound: 82.1186332
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186311
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 1, lower bound: -82.1186309, upper bound: 82.1186332
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 1, lower bound: -82.1186332, upper bound: 82.1186309
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 23.52
Output dim: 1, lower bound: -82.1186311, upper bound: 82.1186332

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962046
time: 8.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962046
time: 8.72 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
time: 10.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
time: 11.01 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
time: 9.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
time: 8.22 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
time: 9.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962011
time: 10.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
time: 9.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
time: 9.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
time: 10.26 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
time: 10.96 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
time: 8.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
time: 9.98 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
time: 9.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962011
time: 10.16 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 21.39 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962046
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962011, upper bound: 82.0962046
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962011
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962015, upper bound: 82.0962042
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 21.39
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962011

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 10.75 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.03 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 9.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 9.45 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 9.43 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 9.44 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.75 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.77 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 8.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 8.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.65 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.24 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 8.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 9.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 8.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
time: 9.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 9.17 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=49, inp2_unstable=49, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=109, inp2_unstable=109, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=251, inp2_unstable=251, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 8.23 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
time: 11.06 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.66 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889269
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 20.66
Output dim: 1, lower bound: -82.0889270, upper bound: 82.0889270
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 20.66
Output dim: 1, lower bound: -82.0962046, upper bound: 82.0962015
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 20.66
Output dim: 1, lower bound: -82.0962042, upper bound: 82.0962011

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 15.31 + 601.04 = 616.34 seconds
