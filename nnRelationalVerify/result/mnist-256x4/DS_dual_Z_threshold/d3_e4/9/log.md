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
execution time: IAR + RelationalAnalysis = 0.84 + 8.06 = 8.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -197.4408316, upper bound: 197.4408316

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4401842, upper bound: 197.4401890
time: 5.96 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4401890, upper bound: 197.4401842
time: 6.56 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 12.59 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 12.59
Output dim: 4, lower bound: -197.4401842, upper bound: 197.4401890
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 12.59
Output dim: 4, lower bound: -197.4401890, upper bound: 197.4401842

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381031, upper bound: 197.4380989
time: 5.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
time: 5.24 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 105
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 105

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
time: 6.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380989, upper bound: 197.4381031
time: 5.04 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 12.45 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.45
Output dim: 4, lower bound: -197.4381031, upper bound: 197.4380989
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.45
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 12.45
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 12.45
Output dim: 4, lower bound: -197.4380989, upper bound: 197.4381031

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

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381031, upper bound: 197.4380989
time: 5.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381030, upper bound: 197.4380987
time: 5.38 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
time: 5.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
time: 5.36 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
time: 5.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
time: 4.64 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380987, upper bound: 197.4381030
time: 5.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380989, upper bound: 197.4381031
time: 5.88 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 12.27 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 4, lower bound: -197.4381031, upper bound: 197.4380989
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 4, lower bound: -197.4381030, upper bound: 197.4380987
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 4, lower bound: -197.4380987, upper bound: 197.4381030
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 12.27
Output dim: 4, lower bound: -197.4380989, upper bound: 197.4381031

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149540, upper bound: 197.3149522
time: 4.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149540, upper bound: 197.3149522
time: 4.47 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149544, upper bound: 197.3149522
time: 5.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149544, upper bound: 197.3149522
time: 5.37 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149523, upper bound: 197.3149535
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149523, upper bound: 197.3149535
time: 4.95 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149520, upper bound: 197.3149530
time: 4.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149520, upper bound: 197.3149530
time: 5.00 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149530, upper bound: 197.3149520
time: 5.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149530, upper bound: 197.3149520
time: 5.08 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149535, upper bound: 197.3149523
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149535, upper bound: 197.3149523
time: 4.79 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149544
time: 4.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149544
time: 4.75 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 181
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 181

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149540
time: 4.86 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149540
time: 4.84 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 10.49 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149540, upper bound: 197.3149522
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149540, upper bound: 197.3149522
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149544, upper bound: 197.3149522
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149544, upper bound: 197.3149522
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149523, upper bound: 197.3149535
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149523, upper bound: 197.3149535
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149520, upper bound: 197.3149530
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149520, upper bound: 197.3149530
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149530, upper bound: 197.3149520
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149530, upper bound: 197.3149520
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149535, upper bound: 197.3149523
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149535, upper bound: 197.3149523
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149544
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149544
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149540
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 10.49
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149540

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037312, upper bound: 197.3037261
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037317, upper bound: 197.3037262
time: 4.69 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037312, upper bound: 197.3037261
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037317, upper bound: 197.3037262
time: 4.50 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037309, upper bound: 197.3037272
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037313, upper bound: 197.3037272
time: 4.86 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037309, upper bound: 197.3037272
time: 4.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037313, upper bound: 197.3037272
time: 5.00 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037271, upper bound: 197.3037293
time: 5.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037287
time: 4.75 seconds

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

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037271, upper bound: 197.3037293
time: 5.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037287
time: 4.74 seconds

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037299
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037287
time: 5.35 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037299
time: 4.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037287
time: 5.48 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037261
time: 4.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037299, upper bound: 197.3037263
time: 5.04 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037261
time: 4.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037299, upper bound: 197.3037263
time: 5.23 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037270
time: 5.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037293, upper bound: 197.3037271
time: 5.18 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037270
time: 5.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037293, upper bound: 197.3037271
time: 5.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

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
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
time: 5.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
time: 4.87 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
time: 5.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
time: 4.85 seconds

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

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312
time: 4.49 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
time: 4.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312
time: 4.35 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 11.66 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037312, upper bound: 197.3037261
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037317, upper bound: 197.3037262
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037312, upper bound: 197.3037261
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037317, upper bound: 197.3037262
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037309, upper bound: 197.3037272
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037313, upper bound: 197.3037272
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037309, upper bound: 197.3037272
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037313, upper bound: 197.3037272
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037271, upper bound: 197.3037293
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037287
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037271, upper bound: 197.3037293
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037287
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037299
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037287
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037299
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037287
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037261
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037299, upper bound: 197.3037263
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037261
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037299, upper bound: 197.3037263
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037270
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037293, upper bound: 197.3037271
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037270
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037293, upper bound: 197.3037271
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 11.66
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037220, upper bound: 197.3037166
time: 4.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
time: 4.88 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037225, upper bound: 197.3037166
time: 4.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
time: 4.45 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037220, upper bound: 197.3037166
time: 4.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
time: 4.94 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037225, upper bound: 197.3037166
time: 4.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
time: 4.42 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037217, upper bound: 197.3037166
time: 4.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037180
time: 5.15 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037222, upper bound: 197.3037166
time: 4.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037181
time: 4.58 seconds

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037217, upper bound: 197.3037166
time: 4.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037180
time: 5.08 seconds

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037222, upper bound: 197.3037166
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037181
time: 4.58 seconds

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037180, upper bound: 197.3037168
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037202
time: 4.45 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037178, upper bound: 197.3037166
time: 4.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037195
time: 4.95 seconds

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037180, upper bound: 197.3037168
time: 4.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037202
time: 4.54 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037178, upper bound: 197.3037166
time: 4.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037195
time: 4.99 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.06 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037170, upper bound: 197.3037173
time: 5.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037207
time: 4.90 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037167
time: 4.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037196
time: 5.02 seconds

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037170, upper bound: 197.3037173
time: 5.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037207
time: 4.82 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037167
time: 4.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037196
time: 5.12 seconds

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037196, upper bound: 197.3037166
time: 4.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037167, upper bound: 197.3037166
time: 4.91 seconds

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

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037207, upper bound: 197.3037166
time: 5.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037173, upper bound: 197.3037170
time: 4.74 seconds

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

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037196, upper bound: 197.3037166
time: 4.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037167, upper bound: 197.3037166
time: 4.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.07 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037207, upper bound: 197.3037166
time: 5.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037173, upper bound: 197.3037170
time: 4.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037195, upper bound: 197.3037166
time: 5.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037178
time: 5.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 197
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 213
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 86
type: DSZ, layer: 1, pos: 196
type: DSZ, layer: 1, pos: 251
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 68
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 226
type: DSZ, layer: 1, pos: 44
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 254
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 64
type: DSZ, layer: 1, pos: 29
type: DSZ, layer: 1, pos: 232
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 128
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 173
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 174
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 233
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 16
type: DSZ, layer: 1, pos: 104
type: DSZ, layer: 1, pos: 138
type: DSZ, layer: 1, pos: 50
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 179
type: DSZ, layer: 1, pos: 155
type: DSZ, layer: 1, pos: 146
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 168
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 108

Time for candidate selection: 0.05 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037202, upper bound: 197.3037166
time: 5.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037168, upper bound: 197.3037180
time: 4.42 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 10.54 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037220, upper bound: 197.3037166
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037225, upper bound: 197.3037166
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037220, upper bound: 197.3037166
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037225, upper bound: 197.3037166
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037217, upper bound: 197.3037166
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037180
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037222, upper bound: 197.3037166
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037181
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037217, upper bound: 197.3037166
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037180
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037222, upper bound: 197.3037166
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037181
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037180, upper bound: 197.3037168
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037202
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037178, upper bound: 197.3037166
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037195
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037180, upper bound: 197.3037168
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037202
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037178, upper bound: 197.3037166
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037195
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037170, upper bound: 197.3037173
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037207
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037167
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037196
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037170, upper bound: 197.3037173
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037207
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037167
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037196
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037196, upper bound: 197.3037166
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037167, upper bound: 197.3037166
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037207, upper bound: 197.3037166
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037173, upper bound: 197.3037170
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037196, upper bound: 197.3037166
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037167, upper bound: 197.3037166
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037207, upper bound: 197.3037166
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037173, upper bound: 197.3037170
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037195, upper bound: 197.3037166
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037178
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037202, upper bound: 197.3037166
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 10.54
Output dim: 4, lower bound: -197.3037168, upper bound: 197.3037180
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.54
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037270
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.54
Output dim: 4, lower bound: -197.3037293, upper bound: 197.3037271
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.54
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.54
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.54
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.54
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.54
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.54
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 10.54
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 10.54
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 8.89 + 599.52 = 608.41 seconds
