## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 197.2433907684


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486)
1: (-87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961)
2: (-114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986)
3: (-122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255)
4: (-112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691)
5: (-100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425)
6: (-96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658)
7: (-105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948)
8: (-125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392)
9: (-96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.89 + 8.24 = 9.13 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -197.4408316, upper bound: 197.4408316

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4408315, upper bound: 197.4408316
time: 6.54 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4408316, upper bound: 197.4408315
time: 5.31 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 11.87 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 11.87
Output dim: 4, lower bound: -197.4408315, upper bound: 197.4408316
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 11.87
Output dim: 4, lower bound: -197.4408316, upper bound: 197.4408315

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 64

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 173

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3719459, upper bound: 197.3719551
time: 5.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3719459, upper bound: 197.3719551
time: 5.00 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3696030, upper bound: 197.3696029
time: 5.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3696030, upper bound: 197.3696029
time: 5.43 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 11.68 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 11.68
Output dim: 4, lower bound: -197.3719459, upper bound: 197.3719551
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 11.68
Output dim: 4, lower bound: -197.3719459, upper bound: 197.3719551
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 11.68
Output dim: 4, lower bound: -197.3696030, upper bound: 197.3696029
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 11.68
Output dim: 4, lower bound: -197.3696030, upper bound: 197.3696029

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3291899, upper bound: 197.3291957
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3291899, upper bound: 197.3291957
time: 6.96 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 179

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3716108, upper bound: 197.3716113
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3716108, upper bound: 197.3716113
time: 7.03 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3134110, upper bound: 197.3134106
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3134110, upper bound: 197.3134106
time: 4.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3256764, upper bound: 197.3256706
time: 4.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3256764, upper bound: 197.3256706
time: 5.02 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 10.73 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.73
Output dim: 4, lower bound: -197.3291899, upper bound: 197.3291957
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.73
Output dim: 4, lower bound: -197.3291899, upper bound: 197.3291957
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.73
Output dim: 4, lower bound: -197.3716108, upper bound: 197.3716113
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.73
Output dim: 4, lower bound: -197.3716108, upper bound: 197.3716113
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.73
Output dim: 4, lower bound: -197.3134110, upper bound: 197.3134106
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.73
Output dim: 4, lower bound: -197.3134110, upper bound: 197.3134106
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 10.73
Output dim: 4, lower bound: -197.3256764, upper bound: 197.3256706
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 10.73
Output dim: 4, lower bound: -197.3256764, upper bound: 197.3256706

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2983982, upper bound: 197.2983993
time: 5.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2983982, upper bound: 197.2983993
time: 5.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3291899, upper bound: 197.3291826
time: 4.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3291820, upper bound: 197.3291957
time: 4.94 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 146

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3401726, upper bound: 197.3401840
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3401726, upper bound: 197.3401840
time: 4.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 197

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3391553, upper bound: 197.3391573
time: 5.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3391553, upper bound: 197.3391573
time: 5.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 168

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3033033, upper bound: 197.3033021
time: 5.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3033022, upper bound: 197.3033022
time: 5.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 16

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3134093, upper bound: 197.3134106
time: 4.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3134110, upper bound: 197.3134084
time: 4.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 105

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 86

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -197.2376618, upper bound: 197.2376620
time: 5.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -197.2376618, upper bound: 197.2376620
time: 5.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3099341, upper bound: 197.3099356
time: 5.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3099341, upper bound: 197.3099356
time: 5.95 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 12.69 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.2983982, upper bound: 197.2983993
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.2983982, upper bound: 197.2983993
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3291899, upper bound: 197.3291826
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3291820, upper bound: 197.3291957
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3401726, upper bound: 197.3401840
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3401726, upper bound: 197.3401840
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3391553, upper bound: 197.3391573
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3391553, upper bound: 197.3391573
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3033033, upper bound: 197.3033021
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3033022, upper bound: 197.3033022
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3134093, upper bound: 197.3134106
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3134110, upper bound: 197.3134084
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.2376618, upper bound: 197.2376620
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.2376618, upper bound: 197.2376620
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3099341, upper bound: 197.3099356
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 12.69
Output dim: 4, lower bound: -197.3099341, upper bound: 197.3099356

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2634768, upper bound: 197.2634765
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2634768, upper bound: 197.2634765
time: 4.54 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2983962, upper bound: 197.2983993
time: 5.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2983982, upper bound: 197.2983972
time: 4.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3005278, upper bound: 197.3005254
time: 4.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3005278, upper bound: 197.3005254
time: 4.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 108

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3154299, upper bound: 197.3154437
time: 5.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3154299, upper bound: 197.3154437
time: 4.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3161317, upper bound: 197.3161375
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3161317, upper bound: 197.3161375
time: 5.02 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 104

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2787587, upper bound: 197.2787599
time: 8.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2787587, upper bound: 197.2787599
time: 7.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 213

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3391553, upper bound: 197.3391497
time: 5.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3391511, upper bound: 197.3391573
time: 4.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2815363, upper bound: 197.2815412
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2815363, upper bound: 197.2815412
time: 5.41 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2863595, upper bound: 197.2863572
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2863595, upper bound: 197.2863572
time: 4.91 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 155

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2932243, upper bound: 197.2932212
time: 4.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2932219, upper bound: 197.2932235
time: 5.57 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2467599, upper bound: 197.2467578
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2467599, upper bound: 197.2467578
time: 5.46 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3133928, upper bound: 197.3133866
time: 5.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3133880, upper bound: 197.3133884
time: 5.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 226

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208
time: 4.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208
time: 5.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208
time: 5.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208
time: 4.99 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 10.89 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2634768, upper bound: 197.2634765
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2634768, upper bound: 197.2634765
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2983962, upper bound: 197.2983993
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2983982, upper bound: 197.2983972
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.3005278, upper bound: 197.3005254
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.3005278, upper bound: 197.3005254
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.3154299, upper bound: 197.3154437
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.3154299, upper bound: 197.3154437
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.3161317, upper bound: 197.3161375
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.3161317, upper bound: 197.3161375
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2787587, upper bound: 197.2787599
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2787587, upper bound: 197.2787599
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.3391553, upper bound: 197.3391497
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.3391511, upper bound: 197.3391573
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2815363, upper bound: 197.2815412
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2815363, upper bound: 197.2815412
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2863595, upper bound: 197.2863572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2863595, upper bound: 197.2863572
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2932243, upper bound: 197.2932212
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2932219, upper bound: 197.2932235
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2467599, upper bound: 197.2467578
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2467599, upper bound: 197.2467578
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.3133928, upper bound: 197.3133866
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.3133880, upper bound: 197.3133884
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.89
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 196

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2634768, upper bound: 197.2634750
time: 5.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2634747, upper bound: 197.2634765
time: 4.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2515345, upper bound: 197.2515364
time: 4.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2515345, upper bound: 197.2515364
time: 5.47 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2983962, upper bound: 197.2983993
time: 5.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2983962, upper bound: 197.2983985
time: 5.17 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 254

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 174

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2594047, upper bound: 197.2594069
time: 5.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2594047, upper bound: 197.2594069
time: 5.12 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 233

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 168

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2727537, upper bound: 197.2727433
time: 4.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2727537, upper bound: 197.2727433
time: 4.60 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3005163, upper bound: 197.3005134
time: 4.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3005164, upper bound: 197.3005140
time: 5.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3154299, upper bound: 197.3154402
time: 5.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3154287, upper bound: 197.3154437
time: 5.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 44

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3119421, upper bound: 197.3119459
time: 5.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3119422, upper bound: 197.3119454
time: 4.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3161317, upper bound: 197.3161332
time: 5.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3161307, upper bound: 197.3161375
time: 5.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 133

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2490782, upper bound: 197.2490768
time: 5.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2490782, upper bound: 197.2490768
time: 4.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.85 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 138

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2787554, upper bound: 197.2787599
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2787587, upper bound: 197.2787559
time: 5.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2742794, upper bound: 197.2742727
time: 4.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2742784, upper bound: 197.2742763
time: 5.65 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 50

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3306688, upper bound: 197.3306675
time: 5.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3306687, upper bound: 197.3306675
time: 5.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 181

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 168

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3203595, upper bound: 197.3203506
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3203595, upper bound: 197.3203506
time: 5.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2815246, upper bound: 197.2815294
time: 5.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2815246, upper bound: 197.2815295
time: 8.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 251

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 4, lower bound: -197.2336372, upper bound: 197.2336358
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 4, lower bound: -197.2336372, upper bound: 197.2336358
time: 5.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 254

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 138

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2863595, upper bound: 197.2863572
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2863594, upper bound: 197.2863538
time: 5.63 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 196

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 128

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2863511, upper bound: 197.2863476
time: 5.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2863501, upper bound: 197.2863483
time: 5.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -104.5059052, 82.9751511, -104.5059052, 82.9751511, -187.4810486, 187.4810486
1: -87.1186676, 73.7036362, -87.1186676, 73.7036362, -160.8222961, 160.8222961
2: -114.9358521, 75.0652771, -114.9358521, 75.0652771, -190.0010986, 190.0010986
3: -122.3928833, 64.3120346, -122.3928833, 64.3120346, -186.7049255, 186.7049255
4: -112.4815369, 86.4718399, -112.4815369, 86.4718399, -198.9533691, 198.9533691
5: -100.3661041, 78.2024307, -100.3661041, 78.2024307, -178.5685425, 178.5685425
6: -96.6764297, 92.3622818, -96.6764297, 92.3622818, -189.0386658, 189.0386658
7: -105.4159775, 88.4643326, -105.4159775, 88.4643326, -193.8802948, 193.8802948
8: -125.9699860, 86.1614532, -125.9699860, 86.1614532, -212.1314392, 212.1314392
9: -96.1728745, 94.4252930, -96.1728745, 94.4252930, -190.5981598, 190.5981598

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 108
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 138

### Candidate
type: DSZ, layer: 1, pos: 29

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2455824, upper bound: 197.2455715
time: 5.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.2455824, upper bound: 197.2455715
time: 5.26 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 13.34 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2634768, upper bound: 197.2634750
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2634747, upper bound: 197.2634765
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2515345, upper bound: 197.2515364
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2515345, upper bound: 197.2515364
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2983962, upper bound: 197.2983993
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2983962, upper bound: 197.2983985
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2594047, upper bound: 197.2594069
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2594047, upper bound: 197.2594069
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2727537, upper bound: 197.2727433
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2727537, upper bound: 197.2727433
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3005163, upper bound: 197.3005134
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3005164, upper bound: 197.3005140
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3154299, upper bound: 197.3154402
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3154287, upper bound: 197.3154437
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3119421, upper bound: 197.3119459
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3119422, upper bound: 197.3119454
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3161317, upper bound: 197.3161332
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3161307, upper bound: 197.3161375
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2490782, upper bound: 197.2490768
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2490782, upper bound: 197.2490768
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2787554, upper bound: 197.2787599
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2787587, upper bound: 197.2787559
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2742794, upper bound: 197.2742727
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2742784, upper bound: 197.2742763
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3306688, upper bound: 197.3306675
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3306687, upper bound: 197.3306675
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3203595, upper bound: 197.3203506
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.3203595, upper bound: 197.3203506
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2815246, upper bound: 197.2815294
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2815246, upper bound: 197.2815295
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2336372, upper bound: 197.2336358
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2336372, upper bound: 197.2336358
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2863595, upper bound: 197.2863572
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2863594, upper bound: 197.2863538
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2863511, upper bound: 197.2863476
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2863501, upper bound: 197.2863483
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2455824, upper bound: 197.2455715
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 13.34
Output dim: 4, lower bound: -197.2455824, upper bound: 197.2455715
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.34
Output dim: 4, lower bound: -197.2932219, upper bound: 197.2932235
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.34
Output dim: 4, lower bound: -197.2467599, upper bound: 197.2467578
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.34
Output dim: 4, lower bound: -197.2467599, upper bound: 197.2467578
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.34
Output dim: 4, lower bound: -197.3133928, upper bound: 197.3133866
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.34
Output dim: 4, lower bound: -197.3133880, upper bound: 197.3133884
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.34
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.34
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 13.34
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 13.34
Output dim: 4, lower bound: -197.2894211, upper bound: 197.2894208

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 9.13 + 600.86 = 609.99 seconds
