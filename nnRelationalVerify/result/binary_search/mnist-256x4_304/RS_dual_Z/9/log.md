## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2000 seconds
Threshold: 197.2433907684
Search space: {k/256 | k = 1, 2, ..., 12}


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

## BASE Result
execution time: IAR + LP analysis = 1.39 + 8.20 = 9.59 seconds
status: Status.UNKNOWN
relational distance
Output dim: 4, lower bound: -197.4409435, upper bound: 197.4409435


# Binary Search by BASE starts (time budget: 1990.41 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=198.953369140625
rel_dist={4: [-197.44087218970873, 197.4408721892934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=198.953369140625
rel_dist={4: [-197.44083159618555, 197.44083163160866]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=198.953369140625
rel_dist={4: [-197.4407374020123, 197.4407374020123]}

## Binary Search Result
Binary search time: 33.22 seconds
BS Status: None
Maximum delta epsilon: None


# Relational Split (RS_dual_Z) starts
Time budget: 1957.19 seconds

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4402161, upper bound: 197.4402243
time: 6.05 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4402243, upper bound: 197.4402161
time: 5.80 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 12.02 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 12.02
Output dim: 4, lower bound: -197.4402161, upper bound: 197.4402243
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 12.02
Output dim: 4, lower bound: -197.4402243, upper bound: 197.4402161

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381547, upper bound: 197.4381430
time: 5.41 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381409, upper bound: 197.4381552
time: 6.56 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381552, upper bound: 197.4381409
time: 5.91 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381430, upper bound: 197.4381547
time: 6.03 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 13.52 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.52
Output dim: 4, lower bound: -197.4381547, upper bound: 197.4381430
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.52
Output dim: 4, lower bound: -197.4381409, upper bound: 197.4381552
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 13.52
Output dim: 4, lower bound: -197.4381552, upper bound: 197.4381409
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 13.52
Output dim: 4, lower bound: -197.4381430, upper bound: 197.4381547

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381547, upper bound: 197.4381430
time: 5.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381545, upper bound: 197.4381426
time: 4.94 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381406, upper bound: 197.4381552
time: 6.47 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381409, upper bound: 197.4381552
time: 6.09 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381552, upper bound: 197.4381409
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381552, upper bound: 197.4381406
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381426, upper bound: 197.4381545
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381430, upper bound: 197.4381547
time: 5.08 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.58 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.58
Output dim: 4, lower bound: -197.4381547, upper bound: 197.4381430
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.58
Output dim: 4, lower bound: -197.4381545, upper bound: 197.4381426
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.58
Output dim: 4, lower bound: -197.4381406, upper bound: 197.4381552
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.58
Output dim: 4, lower bound: -197.4381409, upper bound: 197.4381552
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.58
Output dim: 4, lower bound: -197.4381552, upper bound: 197.4381409
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.58
Output dim: 4, lower bound: -197.4381552, upper bound: 197.4381406
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.58
Output dim: 4, lower bound: -197.4381426, upper bound: 197.4381545
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.58
Output dim: 4, lower bound: -197.4381430, upper bound: 197.4381547

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.50 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149660, upper bound: 197.3149630
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149660, upper bound: 197.3149630
time: 5.20 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.47 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149673, upper bound: 197.3149637
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149673, upper bound: 197.3149637
time: 5.00 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149642, upper bound: 197.3149646
time: 4.99 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149642, upper bound: 197.3149646
time: 4.97 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149632, upper bound: 197.3149640
time: 4.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149632, upper bound: 197.3149640
time: 4.68 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149640, upper bound: 197.3149632
time: 4.50 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149640, upper bound: 197.3149632
time: 4.52 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149646, upper bound: 197.3149642
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149646, upper bound: 197.3149642
time: 4.53 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149637, upper bound: 197.3149673
time: 5.38 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149637, upper bound: 197.3149673
time: 5.40 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149630, upper bound: 197.3149660
time: 5.16 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149630, upper bound: 197.3149660
time: 5.16 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.86 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149660, upper bound: 197.3149630
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149660, upper bound: 197.3149630
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149673, upper bound: 197.3149637
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149673, upper bound: 197.3149637
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149642, upper bound: 197.3149646
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149642, upper bound: 197.3149646
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149632, upper bound: 197.3149640
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149632, upper bound: 197.3149640
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149640, upper bound: 197.3149632
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149640, upper bound: 197.3149632
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149646, upper bound: 197.3149642
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149646, upper bound: 197.3149642
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149637, upper bound: 197.3149673
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149637, upper bound: 197.3149673
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149630, upper bound: 197.3149660
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.86
Output dim: 4, lower bound: -197.3149630, upper bound: 197.3149660

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037351, upper bound: 197.3037261
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037360, upper bound: 197.3037262
time: 4.75 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037351, upper bound: 197.3037261
time: 5.49 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037360, upper bound: 197.3037262
time: 4.82 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037346, upper bound: 197.3037277
time: 5.21 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037355, upper bound: 197.3037275
time: 5.53 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.60 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.18 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037346, upper bound: 197.3037277
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037355, upper bound: 197.3037275
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037274, upper bound: 197.3037320
time: 4.86 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037306
time: 5.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037274, upper bound: 197.3037320
time: 4.87 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037306
time: 5.34 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037329
time: 4.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037307
time: 5.14 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037329
time: 4.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037307
time: 5.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.63 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.19 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037307, upper bound: 197.3037261
time: 5.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037329, upper bound: 197.3037263
time: 4.92 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037307, upper bound: 197.3037261
time: 5.62 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037329, upper bound: 197.3037263
time: 4.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037306, upper bound: 197.3037270
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037320, upper bound: 197.3037274
time: 5.27 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.46 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037306, upper bound: 197.3037270
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037320, upper bound: 197.3037274
time: 5.23 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037275, upper bound: 197.3037355
time: 5.67 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037277, upper bound: 197.3037346
time: 4.83 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037275, upper bound: 197.3037355
time: 5.64 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037277, upper bound: 197.3037346
time: 4.85 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037360
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037351
time: 5.35 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037360
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037351
time: 5.29 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.73 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037351, upper bound: 197.3037261
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037360, upper bound: 197.3037262
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037351, upper bound: 197.3037261
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037360, upper bound: 197.3037262
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037346, upper bound: 197.3037277
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037355, upper bound: 197.3037275
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037346, upper bound: 197.3037277
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037355, upper bound: 197.3037275
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037274, upper bound: 197.3037320
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037306
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037274, upper bound: 197.3037320
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037306
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037329
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037307
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037329
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037307
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037307, upper bound: 197.3037261
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037329, upper bound: 197.3037263
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037307, upper bound: 197.3037261
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037329, upper bound: 197.3037263
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037306, upper bound: 197.3037270
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037320, upper bound: 197.3037274
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037306, upper bound: 197.3037270
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037320, upper bound: 197.3037274
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037275, upper bound: 197.3037355
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037277, upper bound: 197.3037346
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037275, upper bound: 197.3037355
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037277, upper bound: 197.3037346
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037360
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037351
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037360
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.73
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037351

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037259, upper bound: 197.3037166
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037269, upper bound: 197.3037166
time: 4.45 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
time: 5.08 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037259, upper bound: 197.3037166
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
time: 4.74 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037269, upper bound: 197.3037166
time: 4.46 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
time: 5.25 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037255, upper bound: 197.3037166
time: 4.67 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037185
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037264, upper bound: 197.3037166
time: 5.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037183
time: 5.12 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037255, upper bound: 197.3037166
time: 4.63 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037185
time: 5.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037264, upper bound: 197.3037166
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037183
time: 5.18 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.38 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037183, upper bound: 197.3037168
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037229
time: 7.06 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037179, upper bound: 197.3037166
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037215
time: 6.58 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037183, upper bound: 197.3037168
time: 4.88 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037229
time: 7.67 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037179, upper bound: 197.3037166
time: 4.79 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037215
time: 6.55 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037170, upper bound: 197.3037173
time: 5.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037238
time: 5.28 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 14.20 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037259, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037269, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037259, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037269, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037255, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037185
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037264, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037183
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037255, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037185
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037264, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037183
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037183, upper bound: 197.3037168
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037229
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037179, upper bound: 197.3037166
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037215
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037183, upper bound: 197.3037168
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037229
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037179, upper bound: 197.3037166
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037215
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037170, upper bound: 197.3037173
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.20
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037238
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037307
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037329
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037307
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037307, upper bound: 197.3037261
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037329, upper bound: 197.3037263
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037307, upper bound: 197.3037261
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037329, upper bound: 197.3037263
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037306, upper bound: 197.3037270
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037320, upper bound: 197.3037274
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037306, upper bound: 197.3037270
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037320, upper bound: 197.3037274
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037275, upper bound: 197.3037355
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037277, upper bound: 197.3037346
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037275, upper bound: 197.3037355
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037277, upper bound: 197.3037346
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037360
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037351
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037360
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.20
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037351
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=198.953369140625
rel_dist={4: [-197.44087218970873, 197.4408721892934]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4401842, upper bound: 197.4401890
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4401890, upper bound: 197.4401842
time: 7.37 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.13 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 14.13
Output dim: 4, lower bound: -197.4401842, upper bound: 197.4401890
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 14.13
Output dim: 4, lower bound: -197.4401890, upper bound: 197.4401842

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381031, upper bound: 197.4380989
time: 6.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
time: 5.63 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
time: 7.01 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380989, upper bound: 197.4381031
time: 5.52 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 14.04 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.04
Output dim: 4, lower bound: -197.4381031, upper bound: 197.4380989
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.04
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 14.04
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 14.04
Output dim: 4, lower bound: -197.4380989, upper bound: 197.4381031

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381031, upper bound: 197.4380989
time: 5.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381030, upper bound: 197.4380987
time: 5.86 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
time: 6.01 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
time: 5.84 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
time: 6.07 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
time: 5.06 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380987, upper bound: 197.4381030
time: 6.00 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380989, upper bound: 197.4381031
time: 6.28 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 13.75 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 4, lower bound: -197.4381031, upper bound: 197.4380989
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 4, lower bound: -197.4381030, upper bound: 197.4380987
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 4, lower bound: -197.4380973, upper bound: 197.4381042
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 4, lower bound: -197.4381042, upper bound: 197.4380973
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 4, lower bound: -197.4380987, upper bound: 197.4381030
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 13.75
Output dim: 4, lower bound: -197.4380989, upper bound: 197.4381031

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.29 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149540, upper bound: 197.3149522
time: 4.82 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149540, upper bound: 197.3149522
time: 4.85 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149544, upper bound: 197.3149522
time: 5.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149544, upper bound: 197.3149522
time: 5.81 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149523, upper bound: 197.3149535
time: 5.37 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149523, upper bound: 197.3149535
time: 5.35 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149520, upper bound: 197.3149530
time: 5.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149520, upper bound: 197.3149530
time: 5.34 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149530, upper bound: 197.3149520
time: 5.61 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149530, upper bound: 197.3149520
time: 5.67 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149535, upper bound: 197.3149523
time: 5.25 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149535, upper bound: 197.3149523
time: 5.28 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149544
time: 5.22 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149544
time: 5.24 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149540
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149540
time: 5.28 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 12.06 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149540, upper bound: 197.3149522
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149540, upper bound: 197.3149522
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149544, upper bound: 197.3149522
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149544, upper bound: 197.3149522
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149523, upper bound: 197.3149535
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149523, upper bound: 197.3149535
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149520, upper bound: 197.3149530
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149520, upper bound: 197.3149530
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149530, upper bound: 197.3149520
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149530, upper bound: 197.3149520
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149535, upper bound: 197.3149523
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149535, upper bound: 197.3149523
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149544
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149544
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149540
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 12.06
Output dim: 4, lower bound: -197.3149522, upper bound: 197.3149540

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037312, upper bound: 197.3037261
time: 4.84 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037317, upper bound: 197.3037262
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037312, upper bound: 197.3037261
time: 4.83 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037317, upper bound: 197.3037262
time: 4.94 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037309, upper bound: 197.3037272
time: 5.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037313, upper bound: 197.3037272
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037309, upper bound: 197.3037272
time: 5.04 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037313, upper bound: 197.3037272
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037271, upper bound: 197.3037293
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037287
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037271, upper bound: 197.3037293
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037287
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037299
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037287
time: 5.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037299
time: 4.65 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037287
time: 5.74 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.17 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037261
time: 5.24 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037299, upper bound: 197.3037263
time: 5.60 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037261
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037299, upper bound: 197.3037263
time: 5.55 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037270
time: 5.59 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037293, upper bound: 197.3037271
time: 5.48 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037270
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037293, upper bound: 197.3037271
time: 5.50 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
time: 5.16 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
time: 5.54 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
time: 5.20 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
time: 4.77 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312
time: 4.75 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
time: 4.66 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312
time: 4.67 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 13.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037312, upper bound: 197.3037261
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037317, upper bound: 197.3037262
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037312, upper bound: 197.3037261
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037317, upper bound: 197.3037262
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037309, upper bound: 197.3037272
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037313, upper bound: 197.3037272
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037309, upper bound: 197.3037272
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037313, upper bound: 197.3037272
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037271, upper bound: 197.3037293
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037287
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037271, upper bound: 197.3037293
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037270, upper bound: 197.3037287
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037299
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037287
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037299
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037287
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037261
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037299, upper bound: 197.3037263
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037261
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037299, upper bound: 197.3037263
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037270
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037293, upper bound: 197.3037271
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037270
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037293, upper bound: 197.3037271
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 13.12
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037220, upper bound: 197.3037166
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
time: 5.23 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037225, upper bound: 197.3037166
time: 4.71 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
time: 4.68 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037220, upper bound: 197.3037166
time: 4.69 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
time: 5.26 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037225, upper bound: 197.3037166
time: 4.70 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
time: 4.63 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.32 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037217, upper bound: 197.3037166
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037180
time: 5.47 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037222, upper bound: 197.3037166
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037181
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037217, upper bound: 197.3037166
time: 4.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037180
time: 5.46 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037222, upper bound: 197.3037166
time: 4.62 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037181
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037180, upper bound: 197.3037168
time: 5.28 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037202
time: 4.92 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037178, upper bound: 197.3037166
time: 5.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037195
time: 5.36 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.40 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037180, upper bound: 197.3037168
time: 5.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037202
time: 5.04 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.39 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037178, upper bound: 197.3037166
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037195
time: 5.33 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037170, upper bound: 197.3037173
time: 5.33 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037207
time: 5.11 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 14.12 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037220, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037225, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037220, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037177, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037225, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037176, upper bound: 197.3037168
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037217, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037180
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037222, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037181
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037217, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037174, upper bound: 197.3037180
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037222, upper bound: 197.3037166
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037172, upper bound: 197.3037181
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037180, upper bound: 197.3037168
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037202
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037178, upper bound: 197.3037166
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037195
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037180, upper bound: 197.3037168
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037202
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037178, upper bound: 197.3037166
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037195
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037170, upper bound: 197.3037173
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 14.12
Output dim: 4, lower bound: -197.3037166, upper bound: 197.3037207
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037287
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037263, upper bound: 197.3037299
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037287
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037261
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037299, upper bound: 197.3037263
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037261
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037299, upper bound: 197.3037263
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037270
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037293, upper bound: 197.3037271
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037287, upper bound: 197.3037270
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037293, upper bound: 197.3037271
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037313
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037272, upper bound: 197.3037309
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037262, upper bound: 197.3037317
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 14.12
Output dim: 4, lower bound: -197.3037261, upper bound: 197.3037312
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=198.953369140625
rel_dist={4: [-197.44083159618555, 197.44083163160866]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 187

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4401221, upper bound: 197.4401228
time: 8.49 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4401228, upper bound: 197.4401221
time: 7.08 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 15.73 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 15.73
Output dim: 4, lower bound: -197.4401221, upper bound: 197.4401228
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 15.73
Output dim: 4, lower bound: -197.4401228, upper bound: 197.4401221

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380507, upper bound: 197.4380511
time: 7.38 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380507, upper bound: 197.4380531
time: 6.46 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 105

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380531, upper bound: 197.4380507
time: 6.60 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380511, upper bound: 197.4380528
time: 7.54 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 15.60 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 4, lower bound: -197.4380507, upper bound: 197.4380511
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 4, lower bound: -197.4380507, upper bound: 197.4380531
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 4, lower bound: -197.4380531, upper bound: 197.4380507
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 15.60
Output dim: 4, lower bound: -197.4380511, upper bound: 197.4380528

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380528, upper bound: 197.4380511
time: 7.64 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380527, upper bound: 197.4380511
time: 7.96 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380507, upper bound: 197.4380531
time: 8.26 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380507, upper bound: 197.4380531
time: 7.78 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380531, upper bound: 197.4380507
time: 7.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380531, upper bound: 197.4380507
time: 7.79 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380511, upper bound: 197.4380527
time: 7.80 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.4380511, upper bound: 197.4380528
time: 6.63 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 15.88 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.88
Output dim: 4, lower bound: -197.4380528, upper bound: 197.4380511
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.88
Output dim: 4, lower bound: -197.4380527, upper bound: 197.4380511
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.88
Output dim: 4, lower bound: -197.4380507, upper bound: 197.4380531
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.88
Output dim: 4, lower bound: -197.4380507, upper bound: 197.4380531
RS_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.88
Output dim: 4, lower bound: -197.4380531, upper bound: 197.4380507
RS_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.88
Output dim: 4, lower bound: -197.4380531, upper bound: 197.4380507
RS_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 15.88
Output dim: 4, lower bound: -197.4380511, upper bound: 197.4380527
RS_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 15.88
Output dim: 4, lower bound: -197.4380511, upper bound: 197.4380528

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147953, upper bound: 197.3147950
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147953, upper bound: 197.3147950
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.44 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147956, upper bound: 197.3147950
time: 5.44 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147956, upper bound: 197.3147950
time: 5.39 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.28 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147951, upper bound: 197.3147953
time: 5.52 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147951, upper bound: 197.3147953
time: 5.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147949, upper bound: 197.3147951
time: 6.35 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147949, upper bound: 197.3147951
time: 6.31 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.41 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147951, upper bound: 197.3147949
time: 6.21 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147951, upper bound: 197.3147949
time: 6.16 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147953, upper bound: 197.3147951
time: 4.81 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147953, upper bound: 197.3147951
time: 4.81 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.30 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.13 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147950, upper bound: 197.3147956
time: 5.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147950, upper bound: 197.3147956
time: 5.54 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 181
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 181

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147950, upper bound: 197.3147953
time: 5.15 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3147950, upper bound: 197.3147953
time: 5.13 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 11.77 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147953, upper bound: 197.3147950
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147953, upper bound: 197.3147950
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147956, upper bound: 197.3147950
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147956, upper bound: 197.3147950
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147951, upper bound: 197.3147953
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147951, upper bound: 197.3147953
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147949, upper bound: 197.3147951
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147949, upper bound: 197.3147951
RS_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147951, upper bound: 197.3147949
RS_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147951, upper bound: 197.3147949
RS_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147953, upper bound: 197.3147951
RS_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147953, upper bound: 197.3147951
RS_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147950, upper bound: 197.3147956
RS_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147950, upper bound: 197.3147956
RS_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147950, upper bound: 197.3147953
RS_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 11.77
Output dim: 4, lower bound: -197.3147950, upper bound: 197.3147953

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036029, upper bound: 197.3036002
time: 4.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036031, upper bound: 197.3036003
time: 6.41 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036029, upper bound: 197.3036002
time: 4.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036031, upper bound: 197.3036003
time: 6.43 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
time: 6.90 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
time: 6.38 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.31 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
time: 6.42 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
time: 6.24 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
time: 6.50 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.34 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
time: 6.30 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
time: 6.56 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036005, upper bound: 197.3036020
time: 5.66 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036020
time: 6.41 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036005, upper bound: 197.3036020
time: 5.60 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036020
time: 6.40 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036002
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036005
time: 7.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036002
time: 5.10 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036005
time: 7.02 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.16 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
time: 6.04 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
time: 5.99 seconds

## BFS RS instance: RS_RSZ2_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
time: 6.02 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
time: 5.91 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.36 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
time: 38.84 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
time: 6.04 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
time: 38.33 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
time: 6.00 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036003, upper bound: 197.3036031
time: 6.32 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036029
time: 5.01 seconds

## BFS RS instance: RS_RSZ2_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 114
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 114

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036003, upper bound: 197.3036031
time: 6.31 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036029
time: 5.01 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 15.22 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036029, upper bound: 197.3036002
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036031, upper bound: 197.3036003
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036029, upper bound: 197.3036002
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036031, upper bound: 197.3036003
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036005, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036005, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036020
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036002
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036005
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036002
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036005
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036003, upper bound: 197.3036031
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036029
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036003, upper bound: 197.3036031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.22
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036029

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.33 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.15 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3035927, upper bound: 197.3035899
time: 4.93 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3035912, upper bound: 197.3035900
time: 5.14 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.37 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3035929, upper bound: 197.3035899
time: 7.11 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3035913, upper bound: 197.3035901
time: 4.89 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3035927, upper bound: 197.3035899
time: 4.97 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3035912, upper bound: 197.3035900
time: 5.11 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
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

Time for backsubstitution: 1.35 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 93
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 6
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 226
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 64
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 232
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 223
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 249
type: RSZ, layer: 1, pos: 173
type: RSZ, layer: 1, pos: 111
type: RSZ, layer: 1, pos: 46
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 16
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 50
type: RSZ, layer: 1, pos: 21
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 72
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 133
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 155
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 37
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 168
type: RSZ, layer: 1, pos: 25
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 108

Time for candidate selection: 0.14 seconds

### Candidate
type: RSZ, layer: 1, pos: 93

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: RSZ, layer: 1, pos: 11

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3035929, upper bound: 197.3035899
time: 7.08 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 4, lower bound: -197.3035913, upper bound: 197.3035901
time: 4.85 seconds

## Summary of splitting (split count: 5)
- Time for RS candidates: 15.63 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.63
Output dim: 4, lower bound: -197.3035927, upper bound: 197.3035899
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.63
Output dim: 4, lower bound: -197.3035912, upper bound: 197.3035900
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.63
Output dim: 4, lower bound: -197.3035929, upper bound: 197.3035899
RS_RSZ1_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.63
Output dim: 4, lower bound: -197.3035913, upper bound: 197.3035901
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.63
Output dim: 4, lower bound: -197.3035927, upper bound: 197.3035899
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.63
Output dim: 4, lower bound: -197.3035912, upper bound: 197.3035900
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 6, time: 15.63
Output dim: 4, lower bound: -197.3035929, upper bound: 197.3035899
RS_RSZ1_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 6, time: 15.63
Output dim: 4, lower bound: -197.3035913, upper bound: 197.3035901
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
RS_RSZ1_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
RS_RSZ1_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036028, upper bound: 197.3036007
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036005, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036005, upper bound: 197.3036020
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036020
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036002
RS_RSZ2_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036005
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036002
RS_RSZ2_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036005
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
RS_RSZ2_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
RS_RSZ2_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036020, upper bound: 197.3036007
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
RS_RSZ2_RSZ2_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
RS_RSZ2_RSZ2_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036007, upper bound: 197.3036028
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036003, upper bound: 197.3036031
RS_RSZ2_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036029
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036003, upper bound: 197.3036031
RS_RSZ2_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 5, time: 15.63
Output dim: 4, lower bound: -197.3036002, upper bound: 197.3036029
Binary search (step 2): status=Status.UNKNOWN, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=198.953369140625
rel_dist={4: [-197.4407374020123, 197.4407374020123]}

## Binary Search with RS_dual_Z Result
status: None
Maximum delta epsilon: None
execution time: 1818.78 seconds
