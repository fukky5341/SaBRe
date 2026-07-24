## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 189.2309129667


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795)
1: (-85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841)
2: (-113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534)
3: (-120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785)
4: (-110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165)
5: (-99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473)
6: (-95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926)
7: (-103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229)
8: (-124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940)
9: (-94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 2.73 + 10.80 = 13.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -189.4203333, upper bound: 189.4203333

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4140152, upper bound: 189.4140142
time: 7.51 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4140142, upper bound: 189.4140152
time: 7.58 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 15.38 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 15.38
Output dim: 7, lower bound: -189.4140152, upper bound: 189.4140142
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 15.38
Output dim: 7, lower bound: -189.4140142, upper bound: 189.4140152

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4124813, upper bound: 189.4124850
time: 6.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4124879, upper bound: 189.4124803
time: 6.85 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4124803, upper bound: 189.4124879
time: 6.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4124850, upper bound: 189.4124813
time: 6.12 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 15.18 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.18
Output dim: 7, lower bound: -189.4124813, upper bound: 189.4124850
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.18
Output dim: 7, lower bound: -189.4124879, upper bound: 189.4124803
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.18
Output dim: 7, lower bound: -189.4124803, upper bound: 189.4124879
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.18
Output dim: 7, lower bound: -189.4124850, upper bound: 189.4124813

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4122448, upper bound: 189.4122482
time: 7.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4122446, upper bound: 189.4122484
time: 8.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4122517, upper bound: 189.4122436
time: 7.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4122505, upper bound: 189.4122438
time: 7.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4122438, upper bound: 189.4122505
time: 8.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4122436, upper bound: 189.4122517
time: 8.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4122484, upper bound: 189.4122446
time: 7.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4122482, upper bound: 189.4122448
time: 7.14 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 17.05 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -189.4122448, upper bound: 189.4122482
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -189.4122446, upper bound: 189.4122484
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -189.4122517, upper bound: 189.4122436
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -189.4122505, upper bound: 189.4122438
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -189.4122438, upper bound: 189.4122505
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -189.4122436, upper bound: 189.4122517
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -189.4122484, upper bound: 189.4122446
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 17.05
Output dim: 7, lower bound: -189.4122482, upper bound: 189.4122448

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061573, upper bound: 189.4061749
time: 8.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061573, upper bound: 189.4061749
time: 6.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061576, upper bound: 189.4061735
time: 7.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061576, upper bound: 189.4061735
time: 7.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061735, upper bound: 189.4061576
time: 6.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061735, upper bound: 189.4061576
time: 8.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061749, upper bound: 189.4061573
time: 7.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061749, upper bound: 189.4061573
time: 6.75 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061573, upper bound: 189.4061749
time: 7.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061573, upper bound: 189.4061749
time: 10.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061576, upper bound: 189.4061735
time: 6.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061576, upper bound: 189.4061735
time: 6.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061735, upper bound: 189.4061576
time: 7.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061735, upper bound: 189.4061576
time: 6.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.04 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061749, upper bound: 189.4061573
time: 8.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4061749, upper bound: 189.4061573
time: 7.17 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 17.52 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061573, upper bound: 189.4061749
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061573, upper bound: 189.4061749
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061576, upper bound: 189.4061735
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061576, upper bound: 189.4061735
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061735, upper bound: 189.4061576
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061735, upper bound: 189.4061576
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061749, upper bound: 189.4061573
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061749, upper bound: 189.4061573
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061573, upper bound: 189.4061749
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061573, upper bound: 189.4061749
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061576, upper bound: 189.4061735
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061576, upper bound: 189.4061735
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061735, upper bound: 189.4061576
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061735, upper bound: 189.4061576
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061749, upper bound: 189.4061573
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 17.52
Output dim: 7, lower bound: -189.4061749, upper bound: 189.4061573

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017371, upper bound: 189.4017372
time: 7.97 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017360, upper bound: 189.4017391
time: 6.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017371, upper bound: 189.4017372
time: 7.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017360, upper bound: 189.4017391
time: 6.13 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017365, upper bound: 189.4017372
time: 6.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017357, upper bound: 189.4017395
time: 7.06 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017365, upper bound: 189.4017372
time: 6.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017357, upper bound: 189.4017395
time: 6.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017444, upper bound: 189.4017308
time: 6.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017430, upper bound: 189.4017319
time: 7.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017444, upper bound: 189.4017308
time: 6.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017430, upper bound: 189.4017319
time: 7.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017428, upper bound: 189.4017310
time: 6.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017413, upper bound: 189.4017320
time: 6.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017428, upper bound: 189.4017310
time: 5.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017413, upper bound: 189.4017320
time: 5.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017320, upper bound: 189.4017413
time: 6.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017310, upper bound: 189.4017428
time: 6.80 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017320, upper bound: 189.4017413
time: 6.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017310, upper bound: 189.4017428
time: 7.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017319, upper bound: 189.4017430
time: 6.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017308, upper bound: 189.4017444
time: 7.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017319, upper bound: 189.4017430
time: 6.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017308, upper bound: 189.4017444
time: 7.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017395, upper bound: 189.4017357
time: 6.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017365
time: 6.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017395, upper bound: 189.4017357
time: 6.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017365
time: 6.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017391, upper bound: 189.4017360
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017371
time: 6.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017391, upper bound: 189.4017360
time: 6.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017371
time: 6.00 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 14.43 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017371, upper bound: 189.4017372
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017360, upper bound: 189.4017391
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017371, upper bound: 189.4017372
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017360, upper bound: 189.4017391
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017365, upper bound: 189.4017372
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017357, upper bound: 189.4017395
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017365, upper bound: 189.4017372
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017357, upper bound: 189.4017395
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017444, upper bound: 189.4017308
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017430, upper bound: 189.4017319
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017444, upper bound: 189.4017308
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017430, upper bound: 189.4017319
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017428, upper bound: 189.4017310
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017413, upper bound: 189.4017320
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017428, upper bound: 189.4017310
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017413, upper bound: 189.4017320
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017320, upper bound: 189.4017413
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017310, upper bound: 189.4017428
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017320, upper bound: 189.4017413
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017310, upper bound: 189.4017428
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017319, upper bound: 189.4017430
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017308, upper bound: 189.4017444
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017319, upper bound: 189.4017430
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017308, upper bound: 189.4017444
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017395, upper bound: 189.4017357
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017365
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017395, upper bound: 189.4017357
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017365
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017391, upper bound: 189.4017360
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017371
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017391, upper bound: 189.4017360
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 14.43
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017371

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833866, upper bound: 189.3833924
time: 5.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833866, upper bound: 189.3833924
time: 5.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833844, upper bound: 189.3833931
time: 7.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833844, upper bound: 189.3833931
time: 7.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 2.11 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833866, upper bound: 189.3833924
time: 5.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833866, upper bound: 189.3833924
time: 5.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833844, upper bound: 189.3833931
time: 7.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833844, upper bound: 189.3833931
time: 7.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833883, upper bound: 189.3833905
time: 5.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833883, upper bound: 189.3833905
time: 5.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -103.0916901, 81.6635971, -103.0916901, 81.6635971, -184.7552795, 184.7552795
1: -85.8328323, 72.6308670, -85.8328323, 72.6308670, -158.4636841, 158.4636841
2: -113.0277863, 74.4069977, -113.0277863, 74.4069977, -187.4347534, 187.4347534
3: -120.3313828, 64.2784958, -120.3313828, 64.2784958, -184.6098785, 184.6098785
4: -110.0417786, 84.9963455, -110.0417786, 84.9963455, -195.0381165, 195.0381165
5: -99.1850357, 77.4622269, -99.1850357, 77.4622269, -176.6472473, 176.6472473
6: -95.0854187, 90.8557663, -95.0854187, 90.8557663, -185.9411926, 185.9411926
7: -103.3243332, 87.4172974, -103.3243332, 87.4172974, -190.7416229, 190.7416229
8: -124.1503143, 84.7329102, -124.1503143, 84.7329102, -208.8831940, 208.8831940
9: -94.2049332, 92.8833542, -94.2049332, 92.8833542, -187.0882721, 187.0882721

Time for backsubstitution: 1.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833863, upper bound: 189.3833920
time: 6.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3833863, upper bound: 189.3833920
time: 6.66 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 15.02 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833866, upper bound: 189.3833924
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833866, upper bound: 189.3833924
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833844, upper bound: 189.3833931
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833844, upper bound: 189.3833931
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833866, upper bound: 189.3833924
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833866, upper bound: 189.3833924
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833844, upper bound: 189.3833931
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833844, upper bound: 189.3833931
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833883, upper bound: 189.3833905
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833883, upper bound: 189.3833905
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833863, upper bound: 189.3833920
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 15.02
Output dim: 7, lower bound: -189.3833863, upper bound: 189.3833920
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017365, upper bound: 189.4017372
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017357, upper bound: 189.4017395
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017444, upper bound: 189.4017308
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017430, upper bound: 189.4017319
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017444, upper bound: 189.4017308
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017430, upper bound: 189.4017319
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017428, upper bound: 189.4017310
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017413, upper bound: 189.4017320
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017428, upper bound: 189.4017310
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017413, upper bound: 189.4017320
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017320, upper bound: 189.4017413
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017310, upper bound: 189.4017428
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017320, upper bound: 189.4017413
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017310, upper bound: 189.4017428
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017319, upper bound: 189.4017430
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017308, upper bound: 189.4017444
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017319, upper bound: 189.4017430
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017308, upper bound: 189.4017444
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017395, upper bound: 189.4017357
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017365
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017395, upper bound: 189.4017357
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017365
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017391, upper bound: 189.4017360
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017371
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017391, upper bound: 189.4017360
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.02
Output dim: 7, lower bound: -189.4017372, upper bound: 189.4017371

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 13.53 + 594.54 = 608.06 seconds
