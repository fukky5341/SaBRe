## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 11.027958876


## IAR start

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=256
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610)
1: (-5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247)
2: (-6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207)
3: (-7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554)
4: (-7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166)
5: (-6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819)
6: (-6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685)
7: (-7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842)
8: (-7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869)
9: (-6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.38 + 5.79 = 7.17 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -11.1393523, upper bound: 11.1393524

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 140

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1378573, upper bound: 11.1378644
time: 2.99 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1378644, upper bound: 11.1378573
time: 3.30 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 6.31 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 6.31
Output dim: 7, lower bound: -11.1378573, upper bound: 11.1378644
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 6.31
Output dim: 7, lower bound: -11.1378644, upper bound: 11.1378573

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1353924, upper bound: 11.1353927
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1353924, upper bound: 11.1353927
time: 3.20 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=255
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1378140, upper bound: 11.1378573
time: 3.17 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1378644, upper bound: 11.1378114
time: 3.34 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 7.83 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.83
Output dim: 7, lower bound: -11.1353924, upper bound: 11.1353927
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.83
Output dim: 7, lower bound: -11.1353924, upper bound: 11.1353927
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 7.83
Output dim: 7, lower bound: -11.1378140, upper bound: 11.1378573
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 7.83
Output dim: 7, lower bound: -11.1378644, upper bound: 11.1378114

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.5950583, upper bound: 10.5950583
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.5950583, upper bound: 10.5950583
time: 2.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1325965, upper bound: 11.1325979
time: 3.12 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1325965, upper bound: 11.1325979
time: 3.04 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1373776, upper bound: 11.1373990
time: 4.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1373606, upper bound: 11.1374203
time: 3.94 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=254
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 97

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 195

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1378644, upper bound: 11.1378114
time: 3.48 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1378628, upper bound: 11.1378114
time: 3.11 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 7.92 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 7.92
Output dim: 7, lower bound: -10.5950583, upper bound: 10.5950583
RS_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 7.92
Output dim: 7, lower bound: -10.5950583, upper bound: 10.5950583
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.92
Output dim: 7, lower bound: -11.1325965, upper bound: 11.1325979
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.92
Output dim: 7, lower bound: -11.1325965, upper bound: 11.1325979
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.92
Output dim: 7, lower bound: -11.1373776, upper bound: 11.1373990
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.92
Output dim: 7, lower bound: -11.1373606, upper bound: 11.1374203
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 7.92
Output dim: 7, lower bound: -11.1378644, upper bound: 11.1378114
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 7.92
Output dim: 7, lower bound: -11.1378628, upper bound: 11.1378114

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1315927, upper bound: 11.1315958
time: 3.03 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1315927, upper bound: 11.1315958
time: 2.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.43 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1322790, upper bound: 11.1322792
time: 3.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1322790, upper bound: 11.1322792
time: 3.11 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351323, upper bound: 11.1351349
time: 2.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351323, upper bound: 11.1351349
time: 2.79 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 62

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1370973, upper bound: 11.1371478
time: 3.90 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1370976, upper bound: 11.1371438
time: 4.49 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1305207, upper bound: 11.1305196
time: 3.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1305207, upper bound: 11.1305196
time: 3.25 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=253
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1376810, upper bound: 11.1376238
time: 2.92 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1376810, upper bound: 11.1376239
time: 3.33 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 7.59 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1315927, upper bound: 11.1315958
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1315927, upper bound: 11.1315958
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1322790, upper bound: 11.1322792
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1322790, upper bound: 11.1322792
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1351323, upper bound: 11.1351349
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1351323, upper bound: 11.1351349
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1370973, upper bound: 11.1371478
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1370976, upper bound: 11.1371438
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1305207, upper bound: 11.1305196
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1305207, upper bound: 11.1305196
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1376810, upper bound: 11.1376238
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 7.59
Output dim: 7, lower bound: -11.1376810, upper bound: 11.1376239

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1315927, upper bound: 11.1315866
time: 3.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1315838, upper bound: 11.1315958
time: 3.07 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1315841, upper bound: 11.1315828
time: 3.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1315797, upper bound: 11.1315866
time: 2.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1303807, upper bound: 11.1303826
time: 2.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1303810, upper bound: 11.1303824
time: 3.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.4405146, upper bound: 10.4405139
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.4405146, upper bound: 10.4405139
time: 2.65 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1328653, upper bound: 11.1328735
time: 13.27 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1328653, upper bound: 11.1328736
time: 3.07 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.4575450, upper bound: 10.4575450
time: 1.70 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.4575450, upper bound: 10.4575450
time: 1.69 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 7

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1370960, upper bound: 11.1371474
time: 2.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1370973, upper bound: 11.1371478
time: 3.59 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0920751, upper bound: 11.0920966
time: 5.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0920751, upper bound: 11.0920966
time: 3.82 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1305170, upper bound: 11.1305103
time: 2.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1305113, upper bound: 11.1305158
time: 6.74 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9160984, upper bound: 10.9160893
time: 2.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9160984, upper bound: 10.9160893
time: 3.15 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1374689, upper bound: 11.1374216
time: 3.35 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1374781, upper bound: 11.1374182
time: 4.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=252
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1374689, upper bound: 11.1374216
time: 3.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1374781, upper bound: 11.1374182
time: 2.96 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 8.18 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1315927, upper bound: 11.1315866
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1315838, upper bound: 11.1315958
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1315841, upper bound: 11.1315828
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1315797, upper bound: 11.1315866
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1303807, upper bound: 11.1303826
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1303810, upper bound: 11.1303824
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.18
Output dim: 7, lower bound: -10.4405146, upper bound: 10.4405139
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.18
Output dim: 7, lower bound: -10.4405146, upper bound: 10.4405139
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1328653, upper bound: 11.1328735
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1328653, upper bound: 11.1328736
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.18
Output dim: 7, lower bound: -10.4575450, upper bound: 10.4575450
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.18
Output dim: 7, lower bound: -10.4575450, upper bound: 10.4575450
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1370960, upper bound: 11.1371474
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1370973, upper bound: 11.1371478
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.0920751, upper bound: 11.0920966
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.0920751, upper bound: 11.0920966
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1305170, upper bound: 11.1305103
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1305113, upper bound: 11.1305158
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 8.18
Output dim: 7, lower bound: -10.9160984, upper bound: 10.9160893
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 8.18
Output dim: 7, lower bound: -10.9160984, upper bound: 10.9160893
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1374689, upper bound: 11.1374216
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1374781, upper bound: 11.1374182
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1374689, upper bound: 11.1374216
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 8.18
Output dim: 7, lower bound: -11.1374781, upper bound: 11.1374182

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 253

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1313434, upper bound: 11.1313378
time: 4.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1313434, upper bound: 11.1313378
time: 3.72 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1311723, upper bound: 11.1311861
time: 2.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1311723, upper bound: 11.1311861
time: 4.76 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 253

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1259890, upper bound: 11.1259807
time: 2.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1259890, upper bound: 11.1259807
time: 2.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1212999, upper bound: 11.1213323
time: 2.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1212999, upper bound: 11.1213323
time: 3.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0974667, upper bound: 11.0974620
time: 7.54 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0974667, upper bound: 11.0974620
time: 7.16 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1303669, upper bound: 11.1303676
time: 3.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1303669, upper bound: 11.1303676
time: 3.01 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 190

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1278450, upper bound: 11.1278374
time: 3.11 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1278450, upper bound: 11.1278374
time: 3.12 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1210460, upper bound: 11.1210282
time: 3.09 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1210460, upper bound: 11.1210282
time: 3.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 81

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1347126, upper bound: 11.1347580
time: 3.57 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1347126, upper bound: 11.1347580
time: 3.58 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1368829, upper bound: 11.1369370
time: 3.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1368896, upper bound: 11.1369323
time: 2.77 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 177

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0698813, upper bound: 11.0698985
time: 3.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0698794, upper bound: 11.0699047
time: 2.66 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 167

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0291676, upper bound: 11.0291634
time: 3.78 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0291676, upper bound: 11.0291634
time: 2.65 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1305135, upper bound: 11.1305103
time: 2.46 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1305170, upper bound: 11.1305064
time: 2.12 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1292849, upper bound: 11.1292835
time: 2.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1292849, upper bound: 11.1292835
time: 3.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 86

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1349177, upper bound: 11.1348993
time: 2.94 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1349258, upper bound: 11.1348944
time: 3.05 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369741, upper bound: 11.1369304
time: 3.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369741, upper bound: 11.1369304
time: 3.21 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1362893, upper bound: 11.1362657
time: 4.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1362893, upper bound: 11.1362658
time: 3.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=251
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 215

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1358182, upper bound: 11.1357797
time: 2.76 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1358188, upper bound: 11.1357800
time: 3.11 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 7.23 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1313434, upper bound: 11.1313378
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1313434, upper bound: 11.1313378
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1311723, upper bound: 11.1311861
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1311723, upper bound: 11.1311861
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1259890, upper bound: 11.1259807
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1259890, upper bound: 11.1259807
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1212999, upper bound: 11.1213323
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1212999, upper bound: 11.1213323
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.0974667, upper bound: 11.0974620
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.0974667, upper bound: 11.0974620
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1303669, upper bound: 11.1303676
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1303669, upper bound: 11.1303676
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1278450, upper bound: 11.1278374
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1278450, upper bound: 11.1278374
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1210460, upper bound: 11.1210282
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1210460, upper bound: 11.1210282
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1347126, upper bound: 11.1347580
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1347126, upper bound: 11.1347580
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1368829, upper bound: 11.1369370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1368896, upper bound: 11.1369323
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.0698813, upper bound: 11.0698985
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.0698794, upper bound: 11.0699047
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.0291676, upper bound: 11.0291634
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.0291676, upper bound: 11.0291634
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1305135, upper bound: 11.1305103
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1305170, upper bound: 11.1305064
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1292849, upper bound: 11.1292835
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1292849, upper bound: 11.1292835
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1349177, upper bound: 11.1348993
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1349258, upper bound: 11.1348944
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1369741, upper bound: 11.1369304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1369741, upper bound: 11.1369304
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1362893, upper bound: 11.1362657
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1362893, upper bound: 11.1362658
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1358182, upper bound: 11.1357797
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.23
Output dim: 7, lower bound: -11.1358188, upper bound: 11.1357800

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 148

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0095201, upper bound: 11.0095273
time: 2.48 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0095201, upper bound: 11.0095273
time: 2.48 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1312068, upper bound: 11.1311827
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1311889, upper bound: 11.1312011
time: 3.80 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1311577, upper bound: 11.1311861
time: 2.74 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1311723, upper bound: 11.1311693
time: 3.98 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283898, upper bound: 11.1283957
time: 3.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283898, upper bound: 11.1283957
time: 6.79 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 95

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.5432135, upper bound: 10.5432099
time: 2.14 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.5432135, upper bound: 10.5432099
time: 2.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 188

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1259822, upper bound: 11.1259807
time: 3.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1259890, upper bound: 11.1259739
time: 3.82 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 33

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1211679, upper bound: 11.1211883
time: 2.77 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1211538, upper bound: 11.1211955
time: 3.91 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1212999, upper bound: 11.1213288
time: 2.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1212944, upper bound: 11.1213323
time: 4.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 240

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0974667, upper bound: 11.0974558
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0974599, upper bound: 11.0974620
time: 2.54 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0974616, upper bound: 11.0974620
time: 2.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0974667, upper bound: 11.0974556
time: 2.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1293856, upper bound: 11.1293882
time: 4.10 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1293856, upper bound: 11.1293882
time: 3.21 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 131

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1303669, upper bound: 11.1303563
time: 4.18 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1303551, upper bound: 11.1303676
time: 3.03 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 35

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1270973, upper bound: 11.1270799
time: 3.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1270973, upper bound: 11.1270799
time: 2.80 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 219

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1262024, upper bound: 11.1261907
time: 2.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1262024, upper bound: 11.1261907
time: 2.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1210455, upper bound: 11.1210282
time: 2.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1210460, upper bound: 11.1210281
time: 3.18 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 108

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1210442, upper bound: 11.1210282
time: 2.99 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1210440, upper bound: 11.1210282
time: 3.06 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.5656493, upper bound: 10.5656550
time: 1.82 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.5656493, upper bound: 10.5656550
time: 1.81 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 85

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1321261, upper bound: 11.1321621
time: 3.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1321261, upper bound: 11.1321621
time: 2.85 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 27

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1368829, upper bound: 11.1369316
time: 3.52 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1368777, upper bound: 11.1369370
time: 3.13 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 122

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1367758, upper bound: 11.1368186
time: 3.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1367744, upper bound: 11.1368185
time: 2.82 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 251

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0698813, upper bound: 11.0698985
time: 2.69 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0698807, upper bound: 11.0698985
time: 2.44 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0576678, upper bound: 11.0576951
time: 4.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0576553, upper bound: 11.0576971
time: 2.22 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0219996, upper bound: 11.0219913
time: 2.58 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0220014, upper bound: 11.0219915
time: 26.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 89

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0219996, upper bound: 11.0219913
time: 3.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0220014, upper bound: 11.0219915
time: 3.10 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1305134, upper bound: 11.1305103
time: 2.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1305135, upper bound: 11.1305103
time: 2.41 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 97

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1296767, upper bound: 11.1296602
time: 3.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1296767, upper bound: 11.1296602
time: 3.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -6.7478456, 5.7016158, -6.7478456, 5.7016158, -12.4494610, 12.4494610
1: -5.4021134, 4.9384117, -5.4021134, 4.9384117, -10.3405247, 10.3405247
2: -6.9570780, 4.6066432, -6.9570780, 4.6066432, -11.5637207, 11.5637207
3: -7.8119092, 4.0386467, -7.8119092, 4.0386467, -11.8505554, 11.8505554
4: -7.5572267, 5.8959913, -7.5572267, 5.8959913, -13.4532175, 13.4532166
5: -6.5690393, 5.2315445, -6.5690393, 5.2315445, -11.8005829, 11.8005819
6: -6.1139588, 6.4829106, -6.1139588, 6.4829106, -12.5968685, 12.5968685
7: -7.5546770, 4.8859076, -7.5546770, 4.8859076, -12.4405842, 12.4405842
8: -7.5906458, 5.5113425, -7.5906458, 5.5113425, -13.1019850, 13.1019869
9: -6.1002841, 6.2026968, -6.1002841, 6.2026968, -12.3029804, 12.3029804

### Unstable ReLU Count (Linear/Conv2D Layers)
- layer_idx=0, type=LayerType.Linear, total=256, inp1_unstable=41, inp2_unstable=41, delta_unstable=250
- layer_idx=2, type=LayerType.Linear, total=256, inp1_unstable=70, inp2_unstable=70, delta_unstable=256
- layer_idx=4, type=LayerType.Linear, total=256, inp1_unstable=32, inp2_unstable=32, delta_unstable=256
- layer_idx=6, type=LayerType.Linear, total=256, inp1_unstable=247, inp2_unstable=247, delta_unstable=256
- layer_idx=8, type=LayerType.Linear, total=10, inp1_unstable=10, inp2_unstable=10, delta_unstable=10

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 33
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 146

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1292849, upper bound: 11.1292821
time: 3.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1292800, upper bound: 11.1292835
time: 3.22 seconds

## Summary of splitting (split count: 6)
- Time for RS candidates: 7.70 seconds
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0095201, upper bound: 11.0095273
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0095201, upper bound: 11.0095273
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1312068, upper bound: 11.1311827
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1311889, upper bound: 11.1312011
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1311577, upper bound: 11.1311861
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1311723, upper bound: 11.1311693
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1283898, upper bound: 11.1283957
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1283898, upper bound: 11.1283957
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 7, lower bound: -10.5432135, upper bound: 10.5432099
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 7, lower bound: -10.5432135, upper bound: 10.5432099
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1259822, upper bound: 11.1259807
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1259890, upper bound: 11.1259739
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1211679, upper bound: 11.1211883
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1211538, upper bound: 11.1211955
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1212999, upper bound: 11.1213288
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1212944, upper bound: 11.1213323
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0974667, upper bound: 11.0974558
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0974599, upper bound: 11.0974620
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0974616, upper bound: 11.0974620
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0974667, upper bound: 11.0974556
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1293856, upper bound: 11.1293882
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1293856, upper bound: 11.1293882
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1303669, upper bound: 11.1303563
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1303551, upper bound: 11.1303676
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1270973, upper bound: 11.1270799
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1270973, upper bound: 11.1270799
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1262024, upper bound: 11.1261907
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1262024, upper bound: 11.1261907
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1210455, upper bound: 11.1210282
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1210460, upper bound: 11.1210281
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1210442, upper bound: 11.1210282
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1210440, upper bound: 11.1210282
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 7, lower bound: -10.5656493, upper bound: 10.5656550
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 7, lower bound: -10.5656493, upper bound: 10.5656550
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1321261, upper bound: 11.1321621
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1321261, upper bound: 11.1321621
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1368829, upper bound: 11.1369316
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1368777, upper bound: 11.1369370
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1367758, upper bound: 11.1368186
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1367744, upper bound: 11.1368185
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0698813, upper bound: 11.0698985
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0698807, upper bound: 11.0698985
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0576678, upper bound: 11.0576951
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0576553, upper bound: 11.0576971
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0219996, upper bound: 11.0219913
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0220014, upper bound: 11.0219915
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0219996, upper bound: 11.0219913
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.0220014, upper bound: 11.0219915
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1305134, upper bound: 11.1305103
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1305135, upper bound: 11.1305103
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1296767, upper bound: 11.1296602
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1296767, upper bound: 11.1296602
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1292849, upper bound: 11.1292821
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 7, time: 7.70
Output dim: 7, lower bound: -11.1292800, upper bound: 11.1292835
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 7, lower bound: -11.1292849, upper bound: 11.1292835
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 7, lower bound: -11.1349177, upper bound: 11.1348993
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 7, lower bound: -11.1349258, upper bound: 11.1348944
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 7, lower bound: -11.1369741, upper bound: 11.1369304
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 7, lower bound: -11.1369741, upper bound: 11.1369304
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 7, lower bound: -11.1362893, upper bound: 11.1362657
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 7, lower bound: -11.1362893, upper bound: 11.1362658
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 7, lower bound: -11.1358182, upper bound: 11.1357797
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 7.70
Output dim: 7, lower bound: -11.1358188, upper bound: 11.1357800

## RS Result
status: Status.UNKNOWN
execution time: (base) + (rs) = 7.17 + 593.05 = 600.22 seconds
