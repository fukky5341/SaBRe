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
execution time: IAR + RelationalAnalysis = 0.83 + 10.46 = 11.29 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -189.4203333, upper bound: 189.4203333

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 139
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 139

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4101837, upper bound: 189.4101837
time: 6.73 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4101837, upper bound: 189.4101837
time: 6.62 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 13.36 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 13.36
Output dim: 7, lower bound: -189.4101837, upper bound: 189.4101837
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 13.36
Output dim: 7, lower bound: -189.4101837, upper bound: 189.4101837

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3619735, upper bound: 189.3619735
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3619735, upper bound: 189.3619735
time: 5.42 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4095519, upper bound: 189.4095504
time: 6.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4095504, upper bound: 189.4095519
time: 7.62 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 15.25 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.25
Output dim: 7, lower bound: -189.3619735, upper bound: 189.3619735
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.25
Output dim: 7, lower bound: -189.3619735, upper bound: 189.3619735
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.25
Output dim: 7, lower bound: -189.4095519, upper bound: 189.4095504
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.25
Output dim: 7, lower bound: -189.4095504, upper bound: 189.4095519

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3601587, upper bound: 189.3601588
time: 6.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3601587, upper bound: 189.3601588
time: 6.70 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3475708, upper bound: 189.3475708
time: 6.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3475708, upper bound: 189.3475709
time: 6.31 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3877325, upper bound: 189.3877326
time: 5.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3877325, upper bound: 189.3877326
time: 5.86 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4086845, upper bound: 189.4086679
time: 6.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.4086668, upper bound: 189.4086847
time: 7.11 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 16.35 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.35
Output dim: 7, lower bound: -189.3601587, upper bound: 189.3601588
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.35
Output dim: 7, lower bound: -189.3601587, upper bound: 189.3601588
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.35
Output dim: 7, lower bound: -189.3475708, upper bound: 189.3475708
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.35
Output dim: 7, lower bound: -189.3475708, upper bound: 189.3475709
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.35
Output dim: 7, lower bound: -189.3877325, upper bound: 189.3877326
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.35
Output dim: 7, lower bound: -189.3877325, upper bound: 189.3877326
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.35
Output dim: 7, lower bound: -189.4086845, upper bound: 189.4086679
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.35
Output dim: 7, lower bound: -189.4086668, upper bound: 189.4086847

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 134

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3465825, upper bound: 189.3465825
time: 5.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3465825, upper bound: 189.3465825
time: 5.48 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3601587, upper bound: 189.3601518
time: 5.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3601517, upper bound: 189.3601587
time: 5.25 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3422000, upper bound: 189.3422007
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3422007, upper bound: 189.3422000
time: 4.77 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3434428, upper bound: 189.3434489
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3434489, upper bound: 189.3434428
time: 5.45 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3875306, upper bound: 189.3875338
time: 7.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3875338, upper bound: 189.3875308
time: 6.09 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3877221, upper bound: 189.3877173
time: 5.42 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3877169, upper bound: 189.3877220
time: 5.92 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3989587, upper bound: 189.3989416
time: 7.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3989587, upper bound: 189.3989416
time: 5.78 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 27

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3376271, upper bound: 189.3376425
time: 5.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3376271, upper bound: 189.3376425
time: 5.14 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 11.07 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3465825, upper bound: 189.3465825
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3465825, upper bound: 189.3465825
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3601587, upper bound: 189.3601518
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3601517, upper bound: 189.3601587
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3422000, upper bound: 189.3422007
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3422007, upper bound: 189.3422000
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3434428, upper bound: 189.3434489
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3434489, upper bound: 189.3434428
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3875306, upper bound: 189.3875338
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3875338, upper bound: 189.3875308
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3877221, upper bound: 189.3877173
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3877169, upper bound: 189.3877220
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3989587, upper bound: 189.3989416
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3989587, upper bound: 189.3989416
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3376271, upper bound: 189.3376425
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 11.07
Output dim: 7, lower bound: -189.3376271, upper bound: 189.3376425

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3465825, upper bound: 189.3465789
time: 5.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3465789, upper bound: 189.3465825
time: 5.03 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 171

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3341830, upper bound: 189.3341830
time: 5.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3341830, upper bound: 189.3341830
time: 5.40 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3293188, upper bound: 189.3293160
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3293188, upper bound: 189.3293160
time: 4.59 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3592260, upper bound: 189.3592372
time: 6.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3592285, upper bound: 189.3592341
time: 6.02 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3422000, upper bound: 189.3421964
time: 5.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3421954, upper bound: 189.3422007
time: 5.08 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3421946, upper bound: 189.3422000
time: 5.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3422007, upper bound: 189.3421943
time: 5.53 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3402606, upper bound: 189.3402424
time: 5.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3402387, upper bound: 189.3402685
time: 5.10 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3429957, upper bound: 189.3429973
time: 6.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3430045, upper bound: 189.3429960
time: 5.30 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 216

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3670241, upper bound: 189.3670284
time: 5.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3670241, upper bound: 189.3670284
time: 5.31 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3677593, upper bound: 189.3677584
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3677593, upper bound: 189.3677584
time: 4.90 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3723212, upper bound: 189.3723158
time: 5.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3723212, upper bound: 189.3723158
time: 5.49 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3683723, upper bound: 189.3683760
time: 5.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3683723, upper bound: 189.3683760
time: 5.13 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3989559, upper bound: 189.3989416
time: 5.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3989587, upper bound: 189.3989379
time: 6.78 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 86

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3974959, upper bound: 189.3974875
time: 6.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3975100, upper bound: 189.3974777
time: 6.80 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 153

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3299772, upper bound: 189.3299826
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3299772, upper bound: 189.3299826
time: 4.68 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 171

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 197

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3330750, upper bound: 189.3331131
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3330777, upper bound: 189.3331036
time: 6.35 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 20.03 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3465825, upper bound: 189.3465789
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3465789, upper bound: 189.3465825
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3341830, upper bound: 189.3341830
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3341830, upper bound: 189.3341830
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3293188, upper bound: 189.3293160
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3293188, upper bound: 189.3293160
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3592260, upper bound: 189.3592372
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3592285, upper bound: 189.3592341
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3422000, upper bound: 189.3421964
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3421954, upper bound: 189.3422007
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3421946, upper bound: 189.3422000
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3422007, upper bound: 189.3421943
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3402606, upper bound: 189.3402424
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3402387, upper bound: 189.3402685
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3429957, upper bound: 189.3429973
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3430045, upper bound: 189.3429960
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3670241, upper bound: 189.3670284
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3670241, upper bound: 189.3670284
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3677593, upper bound: 189.3677584
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3677593, upper bound: 189.3677584
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3723212, upper bound: 189.3723158
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3723212, upper bound: 189.3723158
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3683723, upper bound: 189.3683760
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3683723, upper bound: 189.3683760
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3989559, upper bound: 189.3989416
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3989587, upper bound: 189.3989379
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3974959, upper bound: 189.3974875
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3975100, upper bound: 189.3974777
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3299772, upper bound: 189.3299826
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3299772, upper bound: 189.3299826
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3330750, upper bound: 189.3331131
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 20.03
Output dim: 7, lower bound: -189.3330777, upper bound: 189.3331036

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 219

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3465825, upper bound: 189.3465757
time: 6.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3465779, upper bound: 189.3465789
time: 5.69 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3295398, upper bound: 189.3295429
time: 4.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3295398, upper bound: 189.3295429
time: 4.80 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3341830, upper bound: 189.3341773
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3341773, upper bound: 189.3341830
time: 5.53 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 216

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3340876, upper bound: 189.3340876
time: 5.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3340876, upper bound: 189.3340876
time: 5.98 seconds

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 34

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3292532, upper bound: 189.3292425
time: 4.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3292436, upper bound: 189.3292504
time: 5.47 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 85

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 219

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3293148, upper bound: 189.3293141
time: 4.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3293170, upper bound: 189.3293138
time: 5.35 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 57

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 134

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3592257, upper bound: 189.3592372
time: 5.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3592260, upper bound: 189.3592370
time: 6.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 190

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3573765, upper bound: 189.3573803
time: 4.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3573765, upper bound: 189.3573809
time: 5.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3412598, upper bound: 189.3412507
time: 4.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3412598, upper bound: 189.3412507
time: 5.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3406699, upper bound: 189.3406752
time: 5.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3406699, upper bound: 189.3406753
time: 5.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3421360, upper bound: 189.3421357
time: 5.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3421348, upper bound: 189.3421413
time: 5.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3422007, upper bound: 189.3421943
time: 6.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3421978, upper bound: 189.3421943
time: 5.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 134
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 7
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 216
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 57
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 190
type: DSZ, layer: 1, pos: 237
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 27
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 153
type: DSZ, layer: 1, pos: 85
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 214
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 219
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 83
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 171
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 34
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 66
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3392110, upper bound: 189.3391983
time: 6.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -189.3392110, upper bound: 189.3391983
time: 5.04 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 12.82 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3465825, upper bound: 189.3465757
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3465779, upper bound: 189.3465789
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3295398, upper bound: 189.3295429
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3295398, upper bound: 189.3295429
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3341830, upper bound: 189.3341773
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3341773, upper bound: 189.3341830
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3340876, upper bound: 189.3340876
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3340876, upper bound: 189.3340876
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3292532, upper bound: 189.3292425
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3292436, upper bound: 189.3292504
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3293148, upper bound: 189.3293141
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3293170, upper bound: 189.3293138
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3592257, upper bound: 189.3592372
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3592260, upper bound: 189.3592370
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3573765, upper bound: 189.3573803
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3573765, upper bound: 189.3573809
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3412598, upper bound: 189.3412507
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3412598, upper bound: 189.3412507
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3406699, upper bound: 189.3406752
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3406699, upper bound: 189.3406753
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3421360, upper bound: 189.3421357
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3421348, upper bound: 189.3421413
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3422007, upper bound: 189.3421943
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3421978, upper bound: 189.3421943
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3392110, upper bound: 189.3391983
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 12.82
Output dim: 7, lower bound: -189.3392110, upper bound: 189.3391983
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3402387, upper bound: 189.3402685
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3429957, upper bound: 189.3429973
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3430045, upper bound: 189.3429960
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3670241, upper bound: 189.3670284
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3670241, upper bound: 189.3670284
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3677593, upper bound: 189.3677584
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3677593, upper bound: 189.3677584
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3723212, upper bound: 189.3723158
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3723212, upper bound: 189.3723158
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3683723, upper bound: 189.3683760
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3683723, upper bound: 189.3683760
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3989559, upper bound: 189.3989416
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3989587, upper bound: 189.3989379
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3974959, upper bound: 189.3974875
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3975100, upper bound: 189.3974777
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3299772, upper bound: 189.3299826
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3299772, upper bound: 189.3299826
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3330750, upper bound: 189.3331131
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 12.82
Output dim: 7, lower bound: -189.3330777, upper bound: 189.3331036

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 11.29 + 588.76 = 600.06 seconds
