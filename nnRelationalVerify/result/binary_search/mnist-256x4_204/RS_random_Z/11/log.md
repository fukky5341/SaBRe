## Execution arguments:
Dataset: Dataset.MNIST
Network: onnx/mnist-net_256x4.onnx
Epsilon: 0.046875
Initial delta epsilon: 12
Time budget: 2700 seconds
Threshold: 460.407499041
Search space: {k/256.0 | k = 1, 2, ..., 12}


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805)
1: (-215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543)
2: (-282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375)
3: (-301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056)
4: (-276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566)
5: (-246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968)
6: (-236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535)
7: (-257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797)
8: (-309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437)
9: (-234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783)

## BASE Result
execution time: IAR + LP analysis = 1.24 + 13.01 = 14.24 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -460.4076444, upper bound: 460.4076444


# Binary Search by BASE starts (time budget: 2685.76 seconds, max iter: 100)

## Binary search (step 0) starts
Candidate k: 6, corresponding eps: 0.0234375


## IAR start
Binary search (step 0): status=Status.UNKNOWN, k_low=1, k_high=12, k_mid=6, eps_mid=0.0234375, abs_max=463.9727783203125
rel_dist={9: [-460.40761108255293, 460.4076110701256]}

## Binary search (step 1) starts
Candidate k: 3, corresponding eps: 0.0117188


## IAR start
Binary search (step 1): status=Status.UNKNOWN, k_low=1, k_high=5, k_mid=3, eps_mid=0.0117188, abs_max=463.9727783203125
rel_dist={9: [-460.40755870218874, 460.4075586754011]}

## Binary search (step 2) starts
Candidate k: 1, corresponding eps: 0.0039062


## IAR start
Binary search (step 2): status=Status.VERIFIED, k_low=1, k_high=2, k_mid=1, eps_mid=0.0039062, abs_max=463.9727783203125
rel_dist={9: [-460.40746283482497, 460.4074627920745]}

## Binary search (step 3) starts
Candidate k: 2, corresponding eps: 0.0078125


## IAR start
Binary search (step 3): status=Status.UNKNOWN, k_low=2, k_high=2, k_mid=2, eps_mid=0.0078125, abs_max=463.9727783203125
rel_dist={9: [-460.4075190791105, 460.4075190850241]}

## Binary Search Result
Binary search time: 71.37 seconds
BS Status: Status.VERIFIED
Maximum delta epsilon: 0.00390625


# Relational Split (RS_random_Z) starts
Time budget: 2614.38 seconds

## Binary search (step 0) starts
Candidate k: 7, corresponding eps: 0.0273438


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 179

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 49

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.3952420, upper bound: 460.3952420
time: 7.14 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.3952420, upper bound: 460.3952420
time: 7.16 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 14.31 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 14.31
Output dim: 9, lower bound: -460.3952420, upper bound: 460.3952420
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 14.31
Output dim: 9, lower bound: -460.3952420, upper bound: 460.3952420
Binary search (step 0): status=Status.VERIFIED, k_low=2, k_high=12, k_mid=7, eps_mid=0.0273438, abs_max=463.9727783203125
rel_dist={9: [-460.4076172501469, 460.40761719415866]}

## Binary search (step 1) starts
Candidate k: 10, corresponding eps: 0.0390625


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 85

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 20

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4073753, upper bound: 460.4073753
time: 11.69 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4073753, upper bound: 460.4073753
time: 10.34 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 22.05 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 22.05
Output dim: 9, lower bound: -460.4073753, upper bound: 460.4073753
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 22.05
Output dim: 9, lower bound: -460.4073753, upper bound: 460.4073753
Binary search (step 1): status=Status.VERIFIED, k_low=8, k_high=12, k_mid=10, eps_mid=0.0390625, abs_max=463.9727783203125
rel_dist={9: [-460.4076338009184, 460.40763368826947]}

## Binary search (step 2) starts
Candidate k: 11, corresponding eps: 0.0429688


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.00 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 128

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 57

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076328, upper bound: 460.4076289
time: 10.43 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076290, upper bound: 460.4076328
time: 10.71 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 21.15 seconds
RS_RSZ1, status: Status.UNKNOWN, split count: 1, time: 21.15
Output dim: 9, lower bound: -460.4076328, upper bound: 460.4076289
RS_RSZ2, status: Status.UNKNOWN, split count: 1, time: 21.15
Output dim: 9, lower bound: -460.4076290, upper bound: 460.4076328

## BFS RS instance: RS_RSZ1

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 247

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 138

### Relational analysis RSZ of RS_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076166, upper bound: 460.4076141
time: 9.40 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076166, upper bound: 460.4076142
time: 9.46 seconds

## BFS RS instance: RS_RSZ2

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 44

### Relational analysis RSZ of RS_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076204, upper bound: 460.4076249
time: 9.34 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076204, upper bound: 460.4076248
time: 9.48 seconds

## Summary of splitting (split count: 1)
- Time for RS candidates: 19.98 seconds
RS_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.98
Output dim: 9, lower bound: -460.4076166, upper bound: 460.4076141
RS_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.98
Output dim: 9, lower bound: -460.4076166, upper bound: 460.4076142
RS_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 2, time: 19.98
Output dim: 9, lower bound: -460.4076204, upper bound: 460.4076249
RS_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 2, time: 19.98
Output dim: 9, lower bound: -460.4076204, upper bound: 460.4076248

## BFS RS instance: RS_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 196

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076166, upper bound: 460.4076141
time: 10.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4076166, upper bound: 460.4076142
time: 10.66 seconds

## BFS RS instance: RS_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 254

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075640, upper bound: 460.4075637
time: 10.80 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075640, upper bound: 460.4075636
time: 10.94 seconds

## BFS RS instance: RS_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 12

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.3922788, upper bound: 460.3922784
time: 6.75 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.3922788, upper bound: 460.3922784
time: 6.77 seconds

## BFS RS instance: RS_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 29

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 166

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4064356, upper bound: 460.4064351
time: 8.55 seconds

### Relational analysis RSZ of RS_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4064356, upper bound: 460.4064350
time: 8.61 seconds

## Summary of splitting (split count: 2)
- Time for RS candidates: 18.32 seconds
RS_RSZ1_RSZ1_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 9, lower bound: -460.4076166, upper bound: 460.4076141
RS_RSZ1_RSZ1_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 9, lower bound: -460.4076166, upper bound: 460.4076142
RS_RSZ1_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 9, lower bound: -460.4075640, upper bound: 460.4075637
RS_RSZ1_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 3, time: 18.32
Output dim: 9, lower bound: -460.4075640, upper bound: 460.4075636
RS_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.32
Output dim: 9, lower bound: -460.3922788, upper bound: 460.3922784
RS_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.32
Output dim: 9, lower bound: -460.3922788, upper bound: 460.3922784
RS_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 3, time: 18.32
Output dim: 9, lower bound: -460.4064356, upper bound: 460.4064351
RS_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 3, time: 18.32
Output dim: 9, lower bound: -460.4064356, upper bound: 460.4064350

## BFS RS instance: RS_RSZ1_RSZ1_RSZ1

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 84

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074217, upper bound: 460.4074198
time: 9.57 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074217, upper bound: 460.4074198
time: 9.10 seconds

## BFS RS instance: RS_RSZ1_RSZ1_RSZ2

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 185

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4025431, upper bound: 460.4025420
time: 6.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ1_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ1_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4025431, upper bound: 460.4025420
time: 6.93 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.17 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 204

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4066231, upper bound: 460.4066215
time: 9.95 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4066231, upper bound: 460.4066215
time: 9.65 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 182

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075368, upper bound: 460.4075366
time: 10.92 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -460.4075368, upper bound: 460.4075365
time: 11.20 seconds

## Summary of splitting (split count: 3)
- Time for RS candidates: 23.29 seconds
RS_RSZ1_RSZ1_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.29
Output dim: 9, lower bound: -460.4074217, upper bound: 460.4074198
RS_RSZ1_RSZ1_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.29
Output dim: 9, lower bound: -460.4074217, upper bound: 460.4074198
RS_RSZ1_RSZ1_RSZ2_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.29
Output dim: 9, lower bound: -460.4025431, upper bound: 460.4025420
RS_RSZ1_RSZ1_RSZ2_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.29
Output dim: 9, lower bound: -460.4025431, upper bound: 460.4025420
RS_RSZ1_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 4, time: 23.29
Output dim: 9, lower bound: -460.4066231, upper bound: 460.4066215
RS_RSZ1_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 4, time: 23.29
Output dim: 9, lower bound: -460.4066231, upper bound: 460.4066215
RS_RSZ1_RSZ2_RSZ2_RSZ1, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 9, lower bound: -460.4075368, upper bound: 460.4075366
RS_RSZ1_RSZ2_RSZ2_RSZ2, status: Status.UNKNOWN, split count: 4, time: 23.29
Output dim: 9, lower bound: -460.4075368, upper bound: 460.4075365

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ1

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.16 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 251

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 174

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4070410, upper bound: 460.4070410
time: 10.17 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4070410, upper bound: 460.4070410
time: 10.23 seconds

## BFS RS instance: RS_RSZ1_RSZ2_RSZ2_RSZ2

### Backsubstitution after applying RS history:
0: -256.7159119, 203.7015533, -256.7159119, 203.7015533, -460.4174805, 460.4174805
1: -215.5436707, 181.1157990, -215.5436707, 181.1157990, -396.6594543, 396.6594543
2: -282.8133240, 182.7726288, -282.8133240, 182.7726288, -465.5859375, 465.5859375
3: -301.2575073, 158.7781830, -301.2575073, 158.7781830, -460.0357056, 460.0357056
4: -276.2066956, 210.6250763, -276.2066956, 210.6250763, -486.8317566, 486.8317566
5: -246.9300537, 191.3716736, -246.9300537, 191.3716736, -438.3016968, 438.3016968
6: -236.3738251, 227.5385132, -236.3738251, 227.5385132, -463.9123535, 463.9123535
7: -257.5447693, 215.6144562, -257.5447693, 215.6144562, -473.1591797, 473.1591797
8: -309.6375427, 210.9121857, -309.6375427, 210.9121857, -520.5497437, 520.5497437
9: -234.0735016, 229.8993225, -234.0735016, 229.8993225, -463.9727783, 463.9727783

Time for backsubstitution: 1.15 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 233
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 213

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 123

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4070751, upper bound: 460.4070741
time: 10.05 seconds

### Relational analysis RSZ of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2

#### Relational analysis RSZ result of RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4070751, upper bound: 460.4070741
time: 9.31 seconds

## Summary of splitting (split count: 4)
- Time for RS candidates: 20.52 seconds
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 9, lower bound: -460.4070410, upper bound: 460.4070410
RS_RSZ1_RSZ2_RSZ2_RSZ1_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 9, lower bound: -460.4070410, upper bound: 460.4070410
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ1, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 9, lower bound: -460.4070751, upper bound: 460.4070741
RS_RSZ1_RSZ2_RSZ2_RSZ2_RSZ2, status: Status.VERIFIED, split count: 5, time: 20.52
Output dim: 9, lower bound: -460.4070751, upper bound: 460.4070741
Binary search (step 2): status=Status.VERIFIED, k_low=11, k_high=12, k_mid=11, eps_mid=0.0429688, abs_max=463.9727783203125
rel_dist={9: [-460.40763907635636, 460.4076391076393]}

## Binary search (step 3) starts
Candidate k: 12, corresponding eps: 0.0468750


## IAR start

## BFS RS instance: RS

Time for backsubstitution: 0.01 seconds

### RS candidates at layer 1
type: RSZ, layer: 1, pos: 245
type: RSZ, layer: 1, pos: 89
type: RSZ, layer: 1, pos: 199
type: RSZ, layer: 1, pos: 83
type: RSZ, layer: 1, pos: 1
type: RSZ, layer: 1, pos: 147
type: RSZ, layer: 1, pos: 84
type: RSZ, layer: 1, pos: 215
type: RSZ, layer: 1, pos: 113
type: RSZ, layer: 1, pos: 128
type: RSZ, layer: 1, pos: 104
type: RSZ, layer: 1, pos: 20
type: RSZ, layer: 1, pos: 251
type: RSZ, layer: 1, pos: 108
type: RSZ, layer: 1, pos: 166
type: RSZ, layer: 1, pos: 187
type: RSZ, layer: 1, pos: 184
type: RSZ, layer: 1, pos: 242
type: RSZ, layer: 1, pos: 76
type: RSZ, layer: 1, pos: 161
type: RSZ, layer: 1, pos: 196
type: RSZ, layer: 1, pos: 85
type: RSZ, layer: 1, pos: 69
type: RSZ, layer: 1, pos: 185
type: RSZ, layer: 1, pos: 159
type: RSZ, layer: 1, pos: 204
type: RSZ, layer: 1, pos: 182
type: RSZ, layer: 1, pos: 175
type: RSZ, layer: 1, pos: 197
type: RSZ, layer: 1, pos: 44
type: RSZ, layer: 1, pos: 123
type: RSZ, layer: 1, pos: 52
type: RSZ, layer: 1, pos: 134
type: RSZ, layer: 1, pos: 254
type: RSZ, layer: 1, pos: 179
type: RSZ, layer: 1, pos: 54
type: RSZ, layer: 1, pos: 13
type: RSZ, layer: 1, pos: 190
type: RSZ, layer: 1, pos: 167
type: RSZ, layer: 1, pos: 138
type: RSZ, layer: 1, pos: 86
type: RSZ, layer: 1, pos: 216
type: RSZ, layer: 1, pos: 240
type: RSZ, layer: 1, pos: 105
type: RSZ, layer: 1, pos: 12
type: RSZ, layer: 1, pos: 29
type: RSZ, layer: 1, pos: 7
type: RSZ, layer: 1, pos: 144
type: RSZ, layer: 1, pos: 49
type: RSZ, layer: 1, pos: 57
type: RSZ, layer: 1, pos: 140
type: RSZ, layer: 1, pos: 174
type: RSZ, layer: 1, pos: 35
type: RSZ, layer: 1, pos: 247
type: RSZ, layer: 1, pos: 188
type: RSZ, layer: 1, pos: 214
type: RSZ, layer: 1, pos: 122
type: RSZ, layer: 1, pos: 176
type: RSZ, layer: 1, pos: 11
type: RSZ, layer: 1, pos: 219
type: RSZ, layer: 1, pos: 36
type: RSZ, layer: 1, pos: 53
type: RSZ, layer: 1, pos: 15
type: RSZ, layer: 1, pos: 68
type: RSZ, layer: 1, pos: 195
type: RSZ, layer: 1, pos: 97
type: RSZ, layer: 1, pos: 213
type: RSZ, layer: 1, pos: 170
type: RSZ, layer: 1, pos: 153
type: RSZ, layer: 1, pos: 146
type: RSZ, layer: 1, pos: 132
type: RSZ, layer: 1, pos: 109
type: RSZ, layer: 1, pos: 233

Time for candidate selection: 0.00 seconds

### Candidate
type: RSZ, layer: 1, pos: 245

### Relational analysis RSZ of RS_RSZ1

#### Relational analysis RSZ result of RS_RSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074347, upper bound: 460.4074347
time: 9.27 seconds

### Relational analysis RSZ of RS_RSZ2

#### Relational analysis RSZ result of RS_RSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -460.4074347, upper bound: 460.4074347
time: 8.26 seconds

## Summary of splitting (split count: 0)
- Time for RS candidates: 17.54 seconds
RS_RSZ1, status: Status.VERIFIED, split count: 1, time: 17.54
Output dim: 9, lower bound: -460.4074347, upper bound: 460.4074347
RS_RSZ2, status: Status.VERIFIED, split count: 1, time: 17.54
Output dim: 9, lower bound: -460.4074347, upper bound: 460.4074347
Binary search (step 3): status=Status.VERIFIED, k_low=12, k_high=12, k_mid=12, eps_mid=0.0468750, abs_max=463.9727783203125
rel_dist={9: [-460.40764437973144, 460.40764437529225]}

## Binary Search with RS_random_Z Result
status: Status.VERIFIED
Maximum delta epsilon: 0.046875
execution time: 374.07 seconds
