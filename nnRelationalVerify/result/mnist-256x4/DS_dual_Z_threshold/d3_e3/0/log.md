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
execution time: IAR + RelationalAnalysis = 2.12 + 5.72 = 7.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -11.1393523, upper bound: 11.1393524

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413
time: 2.67 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413
time: 2.97 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 5.85 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 5.85
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 5.85
Output dim: 7, lower bound: -11.1392413, upper bound: 11.1392413

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

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1382899, upper bound: 11.1382880
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1382899, upper bound: 11.1382880
time: 3.41 seconds

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

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1382880, upper bound: 11.1382899
time: 3.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1382880, upper bound: 11.1382899
time: 3.46 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 8.66 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.66
Output dim: 7, lower bound: -11.1382899, upper bound: 11.1382880
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.66
Output dim: 7, lower bound: -11.1382899, upper bound: 11.1382880
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 8.66
Output dim: 7, lower bound: -11.1382880, upper bound: 11.1382899
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 8.66
Output dim: 7, lower bound: -11.1382880, upper bound: 11.1382899

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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369767, upper bound: 11.1369759
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369768, upper bound: 11.1369753
time: 3.17 seconds

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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369767, upper bound: 11.1369759
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369768, upper bound: 11.1369753
time: 4.19 seconds

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

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369753, upper bound: 11.1369768
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369759, upper bound: 11.1369767
time: 14.60 seconds

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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369753, upper bound: 11.1369768
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1369759, upper bound: 11.1369767
time: 3.54 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 8.74 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 8.74
Output dim: 7, lower bound: -11.1369767, upper bound: 11.1369759
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 8.74
Output dim: 7, lower bound: -11.1369768, upper bound: 11.1369753
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 8.74
Output dim: 7, lower bound: -11.1369767, upper bound: 11.1369759
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 8.74
Output dim: 7, lower bound: -11.1369768, upper bound: 11.1369753
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 8.74
Output dim: 7, lower bound: -11.1369753, upper bound: 11.1369768
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 8.74
Output dim: 7, lower bound: -11.1369759, upper bound: 11.1369767
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 8.74
Output dim: 7, lower bound: -11.1369753, upper bound: 11.1369768
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 8.74
Output dim: 7, lower bound: -11.1369759, upper bound: 11.1369767

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

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319608, upper bound: 11.1319586
time: 4.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319608, upper bound: 11.1319586
time: 4.55 seconds

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

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
time: 3.21 seconds

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

Time for backsubstitution: 2.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319609, upper bound: 11.1319586
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319609, upper bound: 11.1319586
time: 3.17 seconds

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

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
time: 3.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 2.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
time: 5.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
time: 3.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
time: 2.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
time: 2.55 seconds

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

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
time: 4.54 seconds

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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319609
time: 2.98 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 9.36 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319608, upper bound: 11.1319586
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319608, upper bound: 11.1319586
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319609, upper bound: 11.1319586
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319609, upper bound: 11.1319586
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319601, upper bound: 11.1319588
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319588, upper bound: 11.1319601
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319608
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 9.36
Output dim: 7, lower bound: -11.1319586, upper bound: 11.1319609

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

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
time: 2.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
time: 2.66 seconds

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

Time for backsubstitution: 2.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
time: 2.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
time: 2.67 seconds

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

Time for backsubstitution: 1.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
time: 2.59 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
time: 4.61 seconds

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

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
time: 2.87 seconds

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

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
time: 4.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
time: 3.12 seconds

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

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
time: 3.23 seconds

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

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
time: 2.54 seconds

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

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
time: 2.79 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 2.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
time: 2.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
time: 2.54 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
time: 2.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 2.24 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
time: 2.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
time: 2.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
time: 9.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
time: 9.48 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
time: 2.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
time: 2.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 2.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
time: 3.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
time: 2.74 seconds

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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
time: 3.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
time: 3.15 seconds

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

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 253
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 81
type: DSZ, layer: 1, pos: 95
type: DSZ, layer: 1, pos: 215
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 131
type: DSZ, layer: 1, pos: 12
type: DSZ, layer: 1, pos: 195
type: DSZ, layer: 1, pos: 97
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 148
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 33

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
time: 3.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
time: 3.24 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 8.65 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653436, upper bound: 10.9653076
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653302, upper bound: 10.9653137
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653406, upper bound: 10.9653073
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653287, upper bound: 10.9653129
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653129, upper bound: 10.9653287
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653073, upper bound: 10.9653406
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653137, upper bound: 10.9653302
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 8.65
Output dim: 7, lower bound: -10.9653076, upper bound: 10.9653436

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 7.84 + 289.07 = 296.91 seconds
