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
execution time: IAR + RelationalAnalysis = 1.43 + 5.81 = 7.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -11.1393523, upper bound: 11.1393524

# Relational Split (RS) starts

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413
time: 2.70 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413
time: 3.01 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 5.86 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 5.86
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 5.86
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1382899, upper bound: 11.1382880
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1382899, upper bound: 11.1382880
time: 3.51 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1382880, upper bound: 11.1382899
time: 3.08 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1382880, upper bound: 11.1382899
time: 3.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 8.08 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 8.08
Output dim: 7, lower bound: -11.1382899, upper bound: 11.1382880
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 8.08
Output dim: 7, lower bound: -11.1382899, upper bound: 11.1382880
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 8.08
Output dim: 7, lower bound: -11.1382880, upper bound: 11.1382899
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 8.08
Output dim: 7, lower bound: -11.1382880, upper bound: 11.1382899

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369767, upper bound: 11.1369759
time: 3.22 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369768, upper bound: 11.1369753
time: 3.24 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369767, upper bound: 11.1369759
time: 3.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369768, upper bound: 11.1369753
time: 4.30 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369753, upper bound: 11.1369768
time: 3.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369759, upper bound: 11.1369767
time: 15.11 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369753, upper bound: 11.1369768
time: 3.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369759, upper bound: 11.1369767
time: 3.64 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 8.15 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.15
Output dim: 7, lower bound: -11.1369767, upper bound: 11.1369759
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.15
Output dim: 7, lower bound: -11.1369768, upper bound: 11.1369753
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.15
Output dim: 7, lower bound: -11.1369767, upper bound: 11.1369759
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.15
Output dim: 7, lower bound: -11.1369768, upper bound: 11.1369753
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.15
Output dim: 7, lower bound: -11.1369753, upper bound: 11.1369768
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.15
Output dim: 7, lower bound: -11.1369759, upper bound: 11.1369767
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 8.15
Output dim: 7, lower bound: -11.1369753, upper bound: 11.1369768
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 8.15
Output dim: 7, lower bound: -11.1369759, upper bound: 11.1369767

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319608, upper bound: 11.1319586
time: 4.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319608, upper bound: 11.1319586
time: 4.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
time: 3.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
time: 3.28 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319609, upper bound: 11.1319586
time: 3.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319609, upper bound: 11.1319586
time: 3.25 seconds

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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
time: 3.76 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
time: 5.89 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
time: 3.10 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
time: 2.51 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
time: 2.61 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
time: 2.96 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
time: 4.67 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 213

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
time: 4.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319609
time: 3.03 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 8.76 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319608, upper bound: 11.1319586
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319608, upper bound: 11.1319586
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319609, upper bound: 11.1319586
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319609, upper bound: 11.1319586
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 8.76
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319609

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
time: 2.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
time: 2.65 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
time: 2.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
time: 2.66 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
time: 2.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
time: 4.67 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
time: 2.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
time: 2.91 seconds

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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
time: 4.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
time: 3.16 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
time: 3.72 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
time: 3.28 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
time: 2.55 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
time: 2.55 seconds

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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
time: 2.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
time: 2.84 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
time: 2.44 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
time: 2.57 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
time: 2.41 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
time: 2.41 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
time: 2.40 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
time: 2.40 seconds

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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
time: 9.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
time: 9.89 seconds

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
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
time: 3.03 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
time: 3.03 seconds

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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
time: 3.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
time: 2.79 seconds

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

Time for backsubstitution: 1.42 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
time: 3.20 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
time: 3.20 seconds

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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 253
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 177
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 27
type: RSZ, layer: 1, pos: 62
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 81
type: RSZ, layer: 1, pos: 95
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 131
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 107
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 148
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 33

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
time: 3.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
time: 3.22 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 7.94 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 7.94
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436

## RS Result
status: Status.VERIFIED
execution time: (base) + (rs) = 7.24 + 270.62 = 277.87 seconds
