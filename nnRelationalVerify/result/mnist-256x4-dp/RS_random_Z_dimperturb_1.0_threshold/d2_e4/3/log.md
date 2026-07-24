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
execution time: IAR + RelationalAnalysis = 1.35 + 14.03 = 15.38 seconds
status: Status.UNKNOWN
relational distance
Output dim: 1, lower bound: -82.1305969, upper bound: 82.1305969

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 94
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 94

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1304032, upper bound: 82.1304042
time: 11.80 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.1304042, upper bound: 82.1304032
time: 11.78 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 23.59 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 23.59
Output dim: 1, lower bound: -82.1304032, upper bound: 82.1304042
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 23.59
Output dim: 1, lower bound: -82.1304042, upper bound: 82.1304032

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0984681, upper bound: 82.0984687
time: 9.59 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0984681, upper bound: 82.0984687
time: 9.65 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 208
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 218

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 208

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0984687, upper bound: 82.0984681
time: 8.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0984687, upper bound: 82.0984681
time: 8.20 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 17.70 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.70
Output dim: 1, lower bound: -82.0984681, upper bound: 82.0984687
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.70
Output dim: 1, lower bound: -82.0984681, upper bound: 82.0984687
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 17.70
Output dim: 1, lower bound: -82.0984687, upper bound: 82.0984681
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 17.70
Output dim: 1, lower bound: -82.0984687, upper bound: 82.0984681

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
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 4

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0969228, upper bound: 82.0969241
time: 9.29 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0969240, upper bound: 82.0969237
time: 9.97 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0969228, upper bound: 82.0969241
time: 9.32 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0969240, upper bound: 82.0969237
time: 10.14 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0888159, upper bound: 82.0888188
time: 9.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0888159, upper bound: 82.0888188
time: 10.84 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0937666, upper bound: 82.0937708
time: 8.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0937666, upper bound: 82.0937708
time: 7.39 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 16.72 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.72
Output dim: 1, lower bound: -82.0969228, upper bound: 82.0969241
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.72
Output dim: 1, lower bound: -82.0969240, upper bound: 82.0969237
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.72
Output dim: 1, lower bound: -82.0969228, upper bound: 82.0969241
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.72
Output dim: 1, lower bound: -82.0969240, upper bound: 82.0969237
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.72
Output dim: 1, lower bound: -82.0888159, upper bound: 82.0888188
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.72
Output dim: 1, lower bound: -82.0888159, upper bound: 82.0888188
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 16.72
Output dim: 1, lower bound: -82.0937666, upper bound: 82.0937708
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 16.72
Output dim: 1, lower bound: -82.0937666, upper bound: 82.0937708

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 230

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 120

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0963878, upper bound: 82.0963876
time: 10.20 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0963878, upper bound: 82.0963876
time: 9.53 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 173

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0969240, upper bound: 82.0969230
time: 8.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0969222, upper bound: 82.0969237
time: 8.61 seconds

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

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0800828, upper bound: 82.0800828
time: 8.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0800828, upper bound: 82.0800828
time: 8.66 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 176

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0816516, upper bound: 82.0816456
time: 8.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0816516, upper bound: 82.0816456
time: 11.83 seconds

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

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0883982, upper bound: 82.0884013
time: 9.98 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0883983, upper bound: 82.0884012
time: 9.61 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 194

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 52

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 250

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0883982, upper bound: 82.0884013
time: 7.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0883983, upper bound: 82.0884012
time: 8.58 seconds

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
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0937259, upper bound: 82.0937263
time: 6.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0937201, upper bound: 82.0937349
time: 7.66 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 238

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0937666, upper bound: 82.0937633
time: 6.06 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0937613, upper bound: 82.0937708
time: 7.58 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 14.89 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0963878, upper bound: 82.0963876
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0963878, upper bound: 82.0963876
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0969240, upper bound: 82.0969230
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0969222, upper bound: 82.0969237
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0800828, upper bound: 82.0800828
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0800828, upper bound: 82.0800828
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0816516, upper bound: 82.0816456
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0816516, upper bound: 82.0816456
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0883982, upper bound: 82.0884013
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0883983, upper bound: 82.0884012
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0883982, upper bound: 82.0884013
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0883983, upper bound: 82.0884012
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0937259, upper bound: 82.0937263
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0937201, upper bound: 82.0937349
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0937666, upper bound: 82.0937633
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 14.89
Output dim: 1, lower bound: -82.0937613, upper bound: 82.0937708

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 230

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0933812, upper bound: 82.0933821
time: 9.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0933818, upper bound: 82.0933805
time: 8.99 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 238

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 78

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0963871, upper bound: 82.0963879
time: 8.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0963878, upper bound: 82.0963869
time: 8.05 seconds

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
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 218

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0969240, upper bound: 82.0969228
time: 11.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0969238, upper bound: 82.0969230
time: 8.82 seconds

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
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 78

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 60

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0969212, upper bound: 82.0969237
time: 8.13 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0969222, upper bound: 82.0969219
time: 8.59 seconds

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
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0800825, upper bound: 82.0800828
time: 8.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0800828, upper bound: 82.0800821
time: 10.00 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 102

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 178

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0676041, upper bound: 82.0676022
time: 8.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0676041, upper bound: 82.0676022
time: 7.78 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 10

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0768949, upper bound: 82.0768885
time: 9.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0768949, upper bound: 82.0768885
time: 9.85 seconds

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
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 217

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0764470, upper bound: 82.0764401
time: 9.27 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0764469, upper bound: 82.0764384
time: 5.70 seconds

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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 118

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 1, lower bound: -82.0455734, upper bound: 82.0455739
time: 6.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 1, lower bound: -82.0455734, upper bound: 82.0455739
time: 6.23 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 50

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0883945, upper bound: 82.0884012
time: 8.88 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0883983, upper bound: 82.0883998
time: 10.76 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 92

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 50

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0689559, upper bound: 82.0689575
time: 7.74 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0689559, upper bound: 82.0689575
time: 8.25 seconds

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
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 120

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0878972, upper bound: 82.0878991
time: 10.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0878964, upper bound: 82.0879008
time: 10.62 seconds

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

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 136

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 117

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0937259, upper bound: 82.0937185
time: 14.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0937245, upper bound: 82.0937263
time: 8.24 seconds

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

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 238
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 118

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 10

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0937196, upper bound: 82.0937349
time: 8.30 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0937201, upper bound: 82.0937341
time: 9.27 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 178
type: RSZ, layer: 1, pos: 30
type: RSZ, layer: 1, pos: 60
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 78
type: RSZ, layer: 1, pos: 0
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 118
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 250
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 136
type: RSZ, layer: 1, pos: 67
type: RSZ, layer: 1, pos: 102
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 194
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 92
type: RSZ, layer: 1, pos: 120
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 10
type: RSZ, layer: 1, pos: 230
type: RSZ, layer: 1, pos: 42
type: RSZ, layer: 1, pos: 217
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 4
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 218
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 117
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 161

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0923856, upper bound: 82.0923833
time: 10.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 1, lower bound: -82.0923853, upper bound: 82.0923832
time: 8.48 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.15 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0933812, upper bound: 82.0933821
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0933818, upper bound: 82.0933805
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0963871, upper bound: 82.0963879
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0963878, upper bound: 82.0963869
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0969240, upper bound: 82.0969228
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0969238, upper bound: 82.0969230
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0969212, upper bound: 82.0969237
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0969222, upper bound: 82.0969219
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0800825, upper bound: 82.0800828
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0800828, upper bound: 82.0800821
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0676041, upper bound: 82.0676022
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0676041, upper bound: 82.0676022
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0768949, upper bound: 82.0768885
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0768949, upper bound: 82.0768885
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0764470, upper bound: 82.0764401
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0764469, upper bound: 82.0764384
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0455734, upper bound: 82.0455739
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0455734, upper bound: 82.0455739
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0883945, upper bound: 82.0884012
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0883983, upper bound: 82.0883998
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0689559, upper bound: 82.0689575
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0689559, upper bound: 82.0689575
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0878972, upper bound: 82.0878991
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0878964, upper bound: 82.0879008
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0937259, upper bound: 82.0937185
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0937245, upper bound: 82.0937263
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0937196, upper bound: 82.0937349
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0937201, upper bound: 82.0937341
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0923856, upper bound: 82.0923833
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.15
Output dim: 1, lower bound: -82.0923853, upper bound: 82.0923832
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 22.15
Output dim: 1, lower bound: -82.0937613, upper bound: 82.0937708

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 15.38 + 587.55 = 602.94 seconds
