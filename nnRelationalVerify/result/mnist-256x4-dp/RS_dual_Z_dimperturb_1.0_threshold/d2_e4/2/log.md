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
execution time: IAR + RelationalAnalysis = 1.38 + 8.80 = 10.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -14.5592858, upper bound: 14.5592858

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
time: 4.41 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561
time: 4.01 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 8.56 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 8.56
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 8.56
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
time: 7.69 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 144

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561
time: 4.49 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561
time: 6.11 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 11.94 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 11.94
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 11.94
Output dim: 7, lower bound: -14.5584561, upper bound: 14.5584537
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 11.94
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 11.94
Output dim: 7, lower bound: -14.5584537, upper bound: 14.5584561

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456877
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456877
time: 3.38 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.11 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456955, upper bound: 14.5456878
time: 2.96 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456955, upper bound: 14.5456878
time: 3.72 seconds

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

Time for backsubstitution: 1.23 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456878, upper bound: 14.5456955
time: 2.18 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456878, upper bound: 14.5456955
time: 2.17 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456877, upper bound: 14.5456956
time: 10.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -14.5456877, upper bound: 14.5456956
time: 6.80 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.82 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.82
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456877
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.82
Output dim: 7, lower bound: -14.5456956, upper bound: 14.5456877
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.82
Output dim: 7, lower bound: -14.5456955, upper bound: 14.5456878
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.82
Output dim: 7, lower bound: -14.5456955, upper bound: 14.5456878
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.82
Output dim: 7, lower bound: -14.5456878, upper bound: 14.5456955
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.82
Output dim: 7, lower bound: -14.5456878, upper bound: 14.5456955
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.82
Output dim: 7, lower bound: -14.5456877, upper bound: 14.5456956
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.82
Output dim: 7, lower bound: -14.5456877, upper bound: 14.5456956

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440221
time: 3.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440277, upper bound: 14.5440224
time: 3.00 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440221
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440277, upper bound: 14.5440224
time: 3.01 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440281, upper bound: 14.5440220
time: 3.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440223
time: 3.75 seconds

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

Time for backsubstitution: 1.22 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440281, upper bound: 14.5440220
time: 3.09 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440223
time: 5.32 seconds

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440223, upper bound: 14.5440280
time: 2.97 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440220, upper bound: 14.5440281
time: 5.75 seconds

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

Time for backsubstitution: 1.24 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440223, upper bound: 14.5440280
time: 6.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440220, upper bound: 14.5440281
time: 4.29 seconds

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

Time for backsubstitution: 1.26 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440224, upper bound: 14.5440277
time: 2.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440221, upper bound: 14.5440280
time: 2.72 seconds

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
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 77
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 71
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 171
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 15

Time for candidate selection: 0.12 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440224, upper bound: 14.5440277
time: 2.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -14.5440221, upper bound: 14.5440280
time: 2.69 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 8.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440221
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440277, upper bound: 14.5440224
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440221
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440277, upper bound: 14.5440224
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440281, upper bound: 14.5440220
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440223
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440281, upper bound: 14.5440220
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440280, upper bound: 14.5440223
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440223, upper bound: 14.5440280
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440220, upper bound: 14.5440281
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440223, upper bound: 14.5440280
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440220, upper bound: 14.5440281
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440224, upper bound: 14.5440277
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440221, upper bound: 14.5440280
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440224, upper bound: 14.5440277
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 8.20
Output dim: 7, lower bound: -14.5440221, upper bound: 14.5440280

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 10.17 + 159.08 = 169.25 seconds
