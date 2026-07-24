## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 2)
Time budget: 600 seconds
Split limit: 100
Threshold: 14.544726514199999


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500)
1: (-8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229)
2: (-10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675)
3: (-11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024)
4: (-10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131)
5: (-9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238)
6: (-8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851)
7: (-10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482)
8: (-11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781)
9: (-8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.34 + 8.81 = 10.15 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -14.5592858, upper bound: 14.5592858

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 232

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5591295, upper bound: 14.5591295
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5591295, upper bound: 14.5591295
time: 3.77 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.60 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.60
Output dim: 7, lower bound: -14.5591295, upper bound: 14.5591295
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.60
Output dim: 7, lower bound: -14.5591295, upper bound: 14.5591295

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.20 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 71

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5474544, upper bound: 14.5474544
time: 3.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5474544, upper bound: 14.5474544
time: 3.92 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5573504, upper bound: 14.5573504
time: 2.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5573504, upper bound: 14.5573504
time: 2.74 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 6.94 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 7, lower bound: -14.5474544, upper bound: 14.5474544
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 7, lower bound: -14.5474544, upper bound: 14.5474544
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 7, lower bound: -14.5573504, upper bound: 14.5573504
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 6.94
Output dim: 7, lower bound: -14.5573504, upper bound: 14.5573504

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5468355, upper bound: 14.5468355
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5468355, upper bound: 14.5468355
time: 3.15 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5474151, upper bound: 14.5474176
time: 19.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5474176, upper bound: 14.5474151
time: 3.29 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 25

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5555633, upper bound: 14.5555633
time: 4.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5555633, upper bound: 14.5555633
time: 4.30 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5566306, upper bound: 14.5566306
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5566306, upper bound: 14.5566306
time: 6.00 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 11.23 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.23
Output dim: 7, lower bound: -14.5468355, upper bound: 14.5468355
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.23
Output dim: 7, lower bound: -14.5468355, upper bound: 14.5468355
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.23
Output dim: 7, lower bound: -14.5474151, upper bound: 14.5474176
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.23
Output dim: 7, lower bound: -14.5474176, upper bound: 14.5474151
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.23
Output dim: 7, lower bound: -14.5555633, upper bound: 14.5555633
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.23
Output dim: 7, lower bound: -14.5555633, upper bound: 14.5555633
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 11.23
Output dim: 7, lower bound: -14.5566306, upper bound: 14.5566306
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 11.23
Output dim: 7, lower bound: -14.5566306, upper bound: 14.5566306

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5463311, upper bound: 14.5463320
time: 3.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5463311, upper bound: 14.5463320
time: 3.07 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5454342, upper bound: 14.5454337
time: 4.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5454342, upper bound: 14.5454340
time: 3.40 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470907
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470907
time: 4.37 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5474169, upper bound: 14.5474151
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5474176, upper bound: 14.5474151
time: 5.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5475151, upper bound: 14.5475151
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5475151, upper bound: 14.5475151
time: 4.21 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.21 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 184

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5547857, upper bound: 14.5547856
time: 3.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5547857, upper bound: 14.5547857
time: 3.02 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 71

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5566291, upper bound: 14.5566306
time: 8.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5566306, upper bound: 14.5566291
time: 5.97 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5566306, upper bound: 14.5566301
time: 4.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5566301, upper bound: 14.5566306
time: 3.28 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 8.83 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5463311, upper bound: 14.5463320
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5463311, upper bound: 14.5463320
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5454342, upper bound: 14.5454337
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5454342, upper bound: 14.5454340
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470907
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470907
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5474169, upper bound: 14.5474151
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5474176, upper bound: 14.5474151
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5475151, upper bound: 14.5475151
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5475151, upper bound: 14.5475151
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5547857, upper bound: 14.5547856
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5547857, upper bound: 14.5547857
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5566291, upper bound: 14.5566306
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5566306, upper bound: 14.5566291
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5566306, upper bound: 14.5566301
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.83
Output dim: 7, lower bound: -14.5566301, upper bound: 14.5566306

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5463299, upper bound: 14.5463320
time: 3.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5463311, upper bound: 14.5463304
time: 5.02 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5463311, upper bound: 14.5463305
time: 2.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5463303, upper bound: 14.5463320
time: 2.55 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5454342, upper bound: 14.5454337
time: 2.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5454342, upper bound: 14.5454337
time: 3.34 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5441869, upper bound: 14.5441869
time: 2.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5441869, upper bound: 14.5441869
time: 2.78 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470907
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470906
time: 3.24 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5470899, upper bound: 14.5470907
time: 3.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470896
time: 2.85 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 15

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5452482, upper bound: 14.5452474
time: 3.06 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5452482, upper bound: 14.5452474
time: 3.38 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5473651, upper bound: 14.5473629
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5473653, upper bound: 14.5473629
time: 8.28 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 77

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5475151, upper bound: 14.5475151
time: 9.85 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5475151, upper bound: 14.5475151
time: 9.83 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 77

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5337593, upper bound: 14.5337590
time: 2.29 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5337593, upper bound: 14.5337590
time: 3.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5540198, upper bound: 14.5540165
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5540170, upper bound: 14.5540198
time: 15.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5483821, upper bound: 14.5483823
time: 2.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5483821, upper bound: 14.5483823
time: 2.95 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5444754, upper bound: 14.5444758
time: 8.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5444754, upper bound: 14.5444758
time: 10.69 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5564908, upper bound: 14.5564894
time: 3.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5564894, upper bound: 14.5564903
time: 4.67 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5390667, upper bound: 14.5390667
time: 4.43 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5390667, upper bound: 14.5390667
time: 4.42 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 128

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5566296, upper bound: 14.5566306
time: 17.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5566301, upper bound: 14.5566301
time: 3.66 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 22.65 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5463299, upper bound: 14.5463320
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5463311, upper bound: 14.5463304
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5463311, upper bound: 14.5463305
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5463303, upper bound: 14.5463320
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5454342, upper bound: 14.5454337
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5454342, upper bound: 14.5454337
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5441869, upper bound: 14.5441869
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5441869, upper bound: 14.5441869
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470907
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470906
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5470899, upper bound: 14.5470907
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470896
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5452482, upper bound: 14.5452474
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5452482, upper bound: 14.5452474
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5473651, upper bound: 14.5473629
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5473653, upper bound: 14.5473629
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5475151, upper bound: 14.5475151
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5475151, upper bound: 14.5475151
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5337593, upper bound: 14.5337590
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5337593, upper bound: 14.5337590
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5540198, upper bound: 14.5540165
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5540170, upper bound: 14.5540198
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5483821, upper bound: 14.5483823
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5483821, upper bound: 14.5483823
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5444754, upper bound: 14.5444758
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5444754, upper bound: 14.5444758
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5564908, upper bound: 14.5564894
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5564894, upper bound: 14.5564903
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5390667, upper bound: 14.5390667
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5390667, upper bound: 14.5390667
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5566296, upper bound: 14.5566306
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 22.65
Output dim: 7, lower bound: -14.5566301, upper bound: 14.5566301

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 179

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5443374, upper bound: 14.5443374
time: 3.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5443374, upper bound: 14.5443374
time: 3.50 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 57

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5336446, upper bound: 14.5336449
time: 4.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5336446, upper bound: 14.5336449
time: 18.28 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 132

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5461487, upper bound: 14.5461493
time: 15.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5461498, upper bound: 14.5461452
time: 5.15 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5460570, upper bound: 14.5460596
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5460576, upper bound: 14.5460590
time: 5.98 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5448981, upper bound: 14.5448985
time: 3.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5448981, upper bound: 14.5448985
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440071, upper bound: 14.5440043
time: 3.16 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440058, upper bound: 14.5440063
time: 6.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 197

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 104

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5340241, upper bound: 14.5340251
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5340241, upper bound: 14.5340251
time: 3.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5470899, upper bound: 14.5470906
time: 4.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470906
time: 4.03 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 171

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5470899, upper bound: 14.5470906
time: 2.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5470894, upper bound: 14.5470907
time: 10.88 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.25 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5469067, upper bound: 14.5469079
time: 14.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5469079, upper bound: 14.5469074
time: 3.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5450503, upper bound: 14.5450497
time: 3.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5450505, upper bound: 14.5450494
time: 9.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 1

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5452482, upper bound: 14.5452474
time: 2.89 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5452482, upper bound: 14.5452473
time: 23.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.27 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5462434, upper bound: 14.5462435
time: 3.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5462434, upper bound: 14.5462435
time: 3.69 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5468975, upper bound: 14.5468938
time: 4.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5468975, upper bound: 14.5468938
time: 2.62 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -9.7792997, 7.9942503, -9.7792997, 7.9942503, -17.7735500, 17.7735500
1: -8.0821228, 7.1762996, -8.0821228, 7.1762996, -15.2584229, 15.2584229
2: -10.1421604, 6.3774090, -10.1421604, 6.3774090, -16.5195694, 16.5195675
3: -11.8299160, 5.6739855, -11.8299160, 5.6739855, -17.5039024, 17.5039024
4: -10.7803135, 8.5861006, -10.7803135, 8.5861006, -19.3664131, 19.3664131
5: -9.2919331, 7.2536936, -9.2919331, 7.2536936, -16.5456238, 16.5456238
6: -8.7421551, 9.3571310, -8.7421551, 9.3571310, -18.0992813, 18.0992851
7: -10.7748833, 6.7838645, -10.7748833, 6.7838645, -17.5587482, 17.5587482
8: -11.0645905, 7.8223886, -11.0645905, 7.8223886, -18.8869781, 18.8869781
9: -8.9881773, 9.0959854, -8.9881773, 9.0959854, -18.0841618, 18.0841618

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=34, inp2_unstable=34, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=71, inp2_unstable=71, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=52, inp2_unstable=52, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=256, inp2_unstable=256, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 104

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5459770, upper bound: 14.5459737
time: 3.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5459741, upper bound: 14.5459770
time: 25.38 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 29.75 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5443374, upper bound: 14.5443374
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5443374, upper bound: 14.5443374
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5336446, upper bound: 14.5336449
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5336446, upper bound: 14.5336449
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5461487, upper bound: 14.5461493
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5461498, upper bound: 14.5461452
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5460570, upper bound: 14.5460596
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5460576, upper bound: 14.5460590
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5448981, upper bound: 14.5448985
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5448981, upper bound: 14.5448985
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5440071, upper bound: 14.5440043
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5440058, upper bound: 14.5440063
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5340241, upper bound: 14.5340251
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5340241, upper bound: 14.5340251
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5470899, upper bound: 14.5470906
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5470902, upper bound: 14.5470906
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5470899, upper bound: 14.5470906
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5470894, upper bound: 14.5470907
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5469067, upper bound: 14.5469079
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5469079, upper bound: 14.5469074
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5450503, upper bound: 14.5450497
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5450505, upper bound: 14.5450494
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5452482, upper bound: 14.5452474
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5452482, upper bound: 14.5452473
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5462434, upper bound: 14.5462435
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5462434, upper bound: 14.5462435
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5468975, upper bound: 14.5468938
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5468975, upper bound: 14.5468938
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5459770, upper bound: 14.5459737
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 29.75
Output dim: 7, lower bound: -14.5459741, upper bound: 14.5459770
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.75
Output dim: 7, lower bound: -14.5475151, upper bound: 14.5475151
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.75
Output dim: 7, lower bound: -14.5540198, upper bound: 14.5540165
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.75
Output dim: 7, lower bound: -14.5540170, upper bound: 14.5540198
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.75
Output dim: 7, lower bound: -14.5483821, upper bound: 14.5483823
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.75
Output dim: 7, lower bound: -14.5483821, upper bound: 14.5483823
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.75
Output dim: 7, lower bound: -14.5564908, upper bound: 14.5564894
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.75
Output dim: 7, lower bound: -14.5564894, upper bound: 14.5564903
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 29.75
Output dim: 7, lower bound: -14.5566296, upper bound: 14.5566306
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 29.75
Output dim: 7, lower bound: -14.5566301, upper bound: 14.5566301

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 10.15 + 591.86 = 602.02 seconds
