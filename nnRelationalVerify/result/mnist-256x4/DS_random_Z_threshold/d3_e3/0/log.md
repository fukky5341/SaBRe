## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 0)
Time budget: 600 seconds
Split limit: 100
Threshold: 11.027958876


## IAR start

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
execution time: IAR + RelationalAnalysis = 0.85 + 5.58 = 6.43 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -11.1393523, upper bound: 11.1393524

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1393523, upper bound: 11.1393478
time: 3.39 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1393478, upper bound: 11.1393522
time: 3.72 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.12 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.12
Output dim: 7, lower bound: -11.1393523, upper bound: 11.1393478
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.12
Output dim: 7, lower bound: -11.1393478, upper bound: 11.1393522

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1387548, upper bound: 11.1387513
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1387552, upper bound: 11.1387498
time: 2.56 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1315075, upper bound: 11.1315101
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1315075, upper bound: 11.1315101
time: 2.33 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.47 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 7, lower bound: -11.1387548, upper bound: 11.1387513
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 7, lower bound: -11.1387552, upper bound: 11.1387498
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 7, lower bound: -11.1315075, upper bound: 11.1315101
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.47
Output dim: 7, lower bound: -11.1315075, upper bound: 11.1315101

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1377596, upper bound: 11.1377646
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1377596, upper bound: 11.1377646
time: 3.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1370553, upper bound: 11.1370478
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1370553, upper bound: 11.1370478
time: 6.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9618235, upper bound: 10.9618236
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9618235, upper bound: 10.9618236
time: 2.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1314123, upper bound: 11.1314069
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1314054, upper bound: 11.1314132
time: 3.28 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 9.54 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.54
Output dim: 7, lower bound: -11.1377596, upper bound: 11.1377646
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.54
Output dim: 7, lower bound: -11.1377596, upper bound: 11.1377646
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.54
Output dim: 7, lower bound: -11.1370553, upper bound: 11.1370478
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.54
Output dim: 7, lower bound: -11.1370553, upper bound: 11.1370478
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 9.54
Output dim: 7, lower bound: -10.9618235, upper bound: 10.9618236
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 9.54
Output dim: 7, lower bound: -10.9618235, upper bound: 10.9618236
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 9.54
Output dim: 7, lower bound: -11.1314123, upper bound: 11.1314069
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 9.54
Output dim: 7, lower bound: -11.1314054, upper bound: 11.1314132

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1373365, upper bound: 11.1373188
time: 5.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1373136, upper bound: 11.1373355
time: 3.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1359459, upper bound: 11.1359441
time: 4.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1359459, upper bound: 11.1359441
time: 3.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1368200, upper bound: 11.1368132
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1368218, upper bound: 11.1368099
time: 3.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0687273, upper bound: 11.0687268
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0687273, upper bound: 11.0687268
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.8297629, upper bound: 10.8297592
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.8297629, upper bound: 10.8297592
time: 2.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1292629, upper bound: 11.1292800
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1292662, upper bound: 11.1292777
time: 2.27 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 6.16 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.16
Output dim: 7, lower bound: -11.1373365, upper bound: 11.1373188
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.16
Output dim: 7, lower bound: -11.1373136, upper bound: 11.1373355
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.16
Output dim: 7, lower bound: -11.1359459, upper bound: 11.1359441
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.16
Output dim: 7, lower bound: -11.1359459, upper bound: 11.1359441
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.16
Output dim: 7, lower bound: -11.1368200, upper bound: 11.1368132
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.16
Output dim: 7, lower bound: -11.1368218, upper bound: 11.1368099
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.16
Output dim: 7, lower bound: -11.0687273, upper bound: 11.0687268
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.16
Output dim: 7, lower bound: -11.0687273, upper bound: 11.0687268
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 6.16
Output dim: 7, lower bound: -10.8297629, upper bound: 10.8297592
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 6.16
Output dim: 7, lower bound: -10.8297629, upper bound: 10.8297592
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 6.16
Output dim: 7, lower bound: -11.1292629, upper bound: 11.1292800
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 6.16
Output dim: 7, lower bound: -11.1292662, upper bound: 11.1292777

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1372161, upper bound: 11.1371945
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1372151, upper bound: 11.1371945
time: 4.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1373136, upper bound: 11.1373311
time: 4.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1373062, upper bound: 11.1373355
time: 2.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1339386, upper bound: 11.1339336
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1339386, upper bound: 11.1339336
time: 3.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1290688, upper bound: 11.1290686
time: 2.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1290688, upper bound: 11.1290686
time: 2.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1258710, upper bound: 11.1258758
time: 2.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1258710, upper bound: 11.1258758
time: 2.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1368174, upper bound: 11.1368099
time: 4.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1368218, upper bound: 11.1368029
time: 2.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0687273, upper bound: 11.0687202
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0687210, upper bound: 11.0687268
time: 2.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0687264, upper bound: 11.0687268
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0687273, upper bound: 11.0687260
time: 2.55 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1275760, upper bound: 11.1275866
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1275731, upper bound: 11.1275887
time: 2.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 253

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283590, upper bound: 11.1283741
time: 2.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283590, upper bound: 11.1283741
time: 2.73 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 6.46 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1372161, upper bound: 11.1371945
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1372151, upper bound: 11.1371945
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1373136, upper bound: 11.1373311
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1373062, upper bound: 11.1373355
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1339386, upper bound: 11.1339336
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1339386, upper bound: 11.1339336
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1290688, upper bound: 11.1290686
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1290688, upper bound: 11.1290686
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1258710, upper bound: 11.1258758
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1258710, upper bound: 11.1258758
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1368174, upper bound: 11.1368099
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1368218, upper bound: 11.1368029
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.0687273, upper bound: 11.0687202
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.0687210, upper bound: 11.0687268
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.0687264, upper bound: 11.0687268
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.0687273, upper bound: 11.0687260
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1275760, upper bound: 11.1275866
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1275731, upper bound: 11.1275887
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1283590, upper bound: 11.1283741
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 6.46
Output dim: 7, lower bound: -11.1283590, upper bound: 11.1283741

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369921, upper bound: 11.1369691
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369937, upper bound: 11.1369651
time: 3.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369865, upper bound: 11.1369691
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369892, upper bound: 11.1369651
time: 3.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1370931, upper bound: 11.1371209
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1371043, upper bound: 11.1371129
time: 3.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1372885, upper bound: 11.1373355
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1373061, upper bound: 11.1373227
time: 3.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1339386, upper bound: 11.1339185
time: 3.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1339209, upper bound: 11.1339336
time: 3.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1339300, upper bound: 11.1339336
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1339386, upper bound: 11.1339244
time: 2.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1213303, upper bound: 11.1213205
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1213303, upper bound: 11.1213205
time: 2.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1289193, upper bound: 11.1289187
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1289193, upper bound: 11.1289187
time: 2.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 253

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1257835, upper bound: 11.1257852
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1257835, upper bound: 11.1257852
time: 2.97 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1258710, upper bound: 11.1258730
time: 2.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1258706, upper bound: 11.1258758
time: 2.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351887, upper bound: 11.1351849
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1351879, upper bound: 11.1351848
time: 3.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1356880, upper bound: 11.1356654
time: 2.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1356880, upper bound: 11.1356654
time: 3.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0687273, upper bound: 11.0687197
time: 2.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0687249, upper bound: 11.0687202
time: 2.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0686337, upper bound: 11.0686128
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0686136, upper bound: 11.0686394
time: 2.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 33

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0686058, upper bound: 11.0686063
time: 2.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0686079, upper bound: 11.0686024
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0463855, upper bound: 11.0463927
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0463946, upper bound: 11.0463851
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1272936, upper bound: 11.1273085
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1272995, upper bound: 11.1273061
time: 2.53 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 148

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.6473946, upper bound: 10.6474056
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.6473946, upper bound: 10.6474056
time: 2.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 12

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9163658, upper bound: 10.9163660
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9163658, upper bound: 10.9163660
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 7

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283523, upper bound: 11.1283741
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283590, upper bound: 11.1283686
time: 3.13 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 7.60 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1369921, upper bound: 11.1369691
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1369937, upper bound: 11.1369651
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1369865, upper bound: 11.1369691
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1369892, upper bound: 11.1369651
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1370931, upper bound: 11.1371209
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1371043, upper bound: 11.1371129
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1372885, upper bound: 11.1373355
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1373061, upper bound: 11.1373227
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1339386, upper bound: 11.1339185
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1339209, upper bound: 11.1339336
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1339300, upper bound: 11.1339336
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1339386, upper bound: 11.1339244
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1213303, upper bound: 11.1213205
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1213303, upper bound: 11.1213205
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1289193, upper bound: 11.1289187
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1289193, upper bound: 11.1289187
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1257835, upper bound: 11.1257852
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1257835, upper bound: 11.1257852
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1258710, upper bound: 11.1258730
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1258706, upper bound: 11.1258758
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1351887, upper bound: 11.1351849
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1351879, upper bound: 11.1351848
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1356880, upper bound: 11.1356654
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1356880, upper bound: 11.1356654
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.0687273, upper bound: 11.0687197
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.0687249, upper bound: 11.0687202
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.0686337, upper bound: 11.0686128
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.0686136, upper bound: 11.0686394
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.0686058, upper bound: 11.0686063
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.0686079, upper bound: 11.0686024
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.0463855, upper bound: 11.0463927
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.0463946, upper bound: 11.0463851
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1272936, upper bound: 11.1273085
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1272995, upper bound: 11.1273061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.60
Output dim: 7, lower bound: -10.6473946, upper bound: 10.6474056
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.60
Output dim: 7, lower bound: -10.6473946, upper bound: 10.6474056
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 7.60
Output dim: 7, lower bound: -10.9163658, upper bound: 10.9163660
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 7.60
Output dim: 7, lower bound: -10.9163658, upper bound: 10.9163660
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1283523, upper bound: 11.1283741
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 7.60
Output dim: 7, lower bound: -11.1283590, upper bound: 11.1283686

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369896, upper bound: 11.1369691
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369921, upper bound: 11.1369690
time: 3.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369937, upper bound: 11.1369650
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369926, upper bound: 11.1369647
time: 4.33 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1366627, upper bound: 11.1366433
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1366627, upper bound: 11.1366432
time: 3.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 146

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369805, upper bound: 11.1369651
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369892, upper bound: 11.1369587
time: 3.16 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1314430, upper bound: 11.1314627
time: 8.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1314430, upper bound: 11.1314627
time: 10.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.7356673, upper bound: 10.7356654
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.7356673, upper bound: 10.7356654
time: 2.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 81

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1358453, upper bound: 11.1358628
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1358453, upper bound: 11.1358628
time: 3.36 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1324963, upper bound: 11.1325003
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1324963, upper bound: 11.1325003
time: 3.86 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1218727, upper bound: 11.1218689
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1218727, upper bound: 11.1218689
time: 2.15 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1213076, upper bound: 11.1213205
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1213076, upper bound: 11.1213205
time: 2.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9641030, upper bound: 10.9641038
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9641030, upper bound: 10.9641038
time: 1.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1339342, upper bound: 11.1339244
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1339386, upper bound: 11.1339208
time: 2.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1028819, upper bound: 11.1028727
time: 2.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1028819, upper bound: 11.1028727
time: 2.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1213303, upper bound: 11.1213201
time: 2.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1213288, upper bound: 11.1213205
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.8942096, upper bound: 10.8941891
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.8942096, upper bound: 10.8941891
time: 2.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1289193, upper bound: 11.1289106
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1289111, upper bound: 11.1289187
time: 2.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 131

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1257835, upper bound: 11.1257830
time: 2.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1257817, upper bound: 11.1257852
time: 2.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1251025, upper bound: 11.1251060
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1251025, upper bound: 11.1251060
time: 2.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9104768, upper bound: 10.9104833
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9104768, upper bound: 10.9104833
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1233723, upper bound: 11.1233833
time: 2.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1233732, upper bound: 11.1233805
time: 2.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 97

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1346042, upper bound: 11.1346034
time: 2.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1346042, upper bound: 11.1346034
time: 2.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0023091, upper bound: 11.0023073
time: 2.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0023091, upper bound: 11.0023073
time: 2.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 95

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0450690, upper bound: 11.0450766
time: 2.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0450690, upper bound: 11.0450766
time: 2.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 195

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 12

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1356840, upper bound: 11.1356654
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1356880, upper bound: 11.1356544
time: 2.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9390112, upper bound: 10.9390288
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9390112, upper bound: 10.9390288
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 215

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9667468, upper bound: 10.9667543
time: 8.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9667543, upper bound: 10.9667485
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0579774, upper bound: 11.0579355
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0579713, upper bound: 11.0579355
time: 1.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9675982, upper bound: 10.9676266
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9675982, upper bound: 10.9676266
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0686058, upper bound: 11.0686036
time: 2.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0686021, upper bound: 11.0686063
time: 2.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9388964, upper bound: 10.9389420
time: 2.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9388964, upper bound: 10.9389420
time: 2.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0128299, upper bound: 11.0128422
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -11.0128299, upper bound: 11.0128422
time: 2.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 33
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 7

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0463940, upper bound: 11.0463851
time: 2.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.0463946, upper bound: 11.0463835
time: 2.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.1513539, upper bound: 10.1513612
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.1513539, upper bound: 10.1513612
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 95

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1272990, upper bound: 11.1273061
time: 3.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1272995, upper bound: 11.1273044
time: 3.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 148

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283523, upper bound: 11.1283683
time: 3.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1283501, upper bound: 11.1283741
time: 2.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9163658, upper bound: 10.9163596
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9163658, upper bound: 10.9163596
time: 2.00 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.89 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1369896, upper bound: 11.1369691
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1369921, upper bound: 11.1369690
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1369937, upper bound: 11.1369650
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1369926, upper bound: 11.1369647
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1366627, upper bound: 11.1366433
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1366627, upper bound: 11.1366432
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1369805, upper bound: 11.1369651
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1369892, upper bound: 11.1369587
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1314430, upper bound: 11.1314627
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1314430, upper bound: 11.1314627
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.7356673, upper bound: 10.7356654
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.7356673, upper bound: 10.7356654
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1358453, upper bound: 11.1358628
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1358453, upper bound: 11.1358628
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1324963, upper bound: 11.1325003
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1324963, upper bound: 11.1325003
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1218727, upper bound: 11.1218689
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1218727, upper bound: 11.1218689
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1213076, upper bound: 11.1213205
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1213076, upper bound: 11.1213205
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9641030, upper bound: 10.9641038
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9641030, upper bound: 10.9641038
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1339342, upper bound: 11.1339244
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1339386, upper bound: 11.1339208
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1028819, upper bound: 11.1028727
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1028819, upper bound: 11.1028727
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1213303, upper bound: 11.1213201
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1213288, upper bound: 11.1213205
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.8942096, upper bound: 10.8941891
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.8942096, upper bound: 10.8941891
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1289193, upper bound: 11.1289106
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1289111, upper bound: 11.1289187
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1257835, upper bound: 11.1257830
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1257817, upper bound: 11.1257852
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1251025, upper bound: 11.1251060
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1251025, upper bound: 11.1251060
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9104768, upper bound: 10.9104833
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9104768, upper bound: 10.9104833
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1233723, upper bound: 11.1233833
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1233732, upper bound: 11.1233805
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1346042, upper bound: 11.1346034
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1346042, upper bound: 11.1346034
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0023091, upper bound: 11.0023073
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0023091, upper bound: 11.0023073
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0450690, upper bound: 11.0450766
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0450690, upper bound: 11.0450766
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1356840, upper bound: 11.1356654
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1356880, upper bound: 11.1356544
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9390112, upper bound: 10.9390288
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9390112, upper bound: 10.9390288
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9667468, upper bound: 10.9667543
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9667543, upper bound: 10.9667485
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0579774, upper bound: 11.0579355
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0579713, upper bound: 11.0579355
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9675982, upper bound: 10.9676266
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9675982, upper bound: 10.9676266
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0686058, upper bound: 11.0686036
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0686021, upper bound: 11.0686063
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9388964, upper bound: 10.9389420
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9388964, upper bound: 10.9389420
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0128299, upper bound: 11.0128422
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0128299, upper bound: 11.0128422
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0463940, upper bound: 11.0463851
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.0463946, upper bound: 11.0463835
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.1513539, upper bound: 10.1513612
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.1513539, upper bound: 10.1513612
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1272990, upper bound: 11.1273061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1272995, upper bound: 11.1273044
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1283523, upper bound: 11.1283683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.89
Output dim: 7, lower bound: -11.1283501, upper bound: 11.1283741
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9163658, upper bound: 10.9163596
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.89
Output dim: 7, lower bound: -10.9163658, upper bound: 10.9163596

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 122

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 190

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1322676, upper bound: 11.1322571
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1322676, upper bound: 11.1322571
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369921, upper bound: 11.1369656
time: 4.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369918, upper bound: 11.1369690
time: 4.23 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1367552, upper bound: 11.1367257
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1367573, upper bound: 11.1367087
time: 2.66 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 6.18 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.18
Output dim: 7, lower bound: -11.1322676, upper bound: 11.1322571
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.18
Output dim: 7, lower bound: -11.1322676, upper bound: 11.1322571
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.18
Output dim: 7, lower bound: -11.1369921, upper bound: 11.1369656
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.18
Output dim: 7, lower bound: -11.1369918, upper bound: 11.1369690
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.18
Output dim: 7, lower bound: -11.1367552, upper bound: 11.1367257
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.18
Output dim: 7, lower bound: -11.1367573, upper bound: 11.1367087
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1369926, upper bound: 11.1369647
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1366627, upper bound: 11.1366433
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1366627, upper bound: 11.1366432
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1369805, upper bound: 11.1369651
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1369892, upper bound: 11.1369587
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1314430, upper bound: 11.1314627
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1314430, upper bound: 11.1314627
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1358453, upper bound: 11.1358628
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1358453, upper bound: 11.1358628
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1324963, upper bound: 11.1325003
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1324963, upper bound: 11.1325003
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1218727, upper bound: 11.1218689
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1218727, upper bound: 11.1218689
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1213076, upper bound: 11.1213205
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1213076, upper bound: 11.1213205
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1339342, upper bound: 11.1339244
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1339386, upper bound: 11.1339208
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1028819, upper bound: 11.1028727
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1028819, upper bound: 11.1028727
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1213303, upper bound: 11.1213201
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1213288, upper bound: 11.1213205
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1289193, upper bound: 11.1289106
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1289111, upper bound: 11.1289187
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1257835, upper bound: 11.1257830
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1257817, upper bound: 11.1257852
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1251025, upper bound: 11.1251060
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1251025, upper bound: 11.1251060
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1233723, upper bound: 11.1233833
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1233732, upper bound: 11.1233805
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1346042, upper bound: 11.1346034
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1346042, upper bound: 11.1346034
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.0450690, upper bound: 11.0450766
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.0450690, upper bound: 11.0450766
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1356840, upper bound: 11.1356654
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1356880, upper bound: 11.1356544
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.0579774, upper bound: 11.0579355
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.0579713, upper bound: 11.0579355
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.0686058, upper bound: 11.0686036
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.0686021, upper bound: 11.0686063
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.0463940, upper bound: 11.0463851
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.0463946, upper bound: 11.0463835
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1272990, upper bound: 11.1273061
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1272995, upper bound: 11.1273044
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1283523, upper bound: 11.1283683
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.18
Output dim: 7, lower bound: -11.1283501, upper bound: 11.1283741

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 6.43 + 595.05 = 601.48 seconds
