## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 3)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.00157248


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0062809, 0.0062809)
1: (-0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017708, 0.0017708)
2: (-0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0130656, 0.0130656)
3: (0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017290, 0.0017290)
4: (-0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0097644, 0.0097644)
5: (0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0027128, 0.0027128)
6: (0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024624, 0.0024624)
7: (-0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0091894, 0.0091894)
8: (-0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0071521, 0.0071521)
9: (-0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006171, 0.0006171)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.77 + 2.75 = 3.53 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0017475, upper bound: 0.0017472

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 122
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 122

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0017339, upper bound: 0.0017343
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0017347, upper bound: 0.0017336
time: 1.66 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.49 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.49
Output dim: 5, lower bound: -0.0017339, upper bound: 0.0017343
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.49
Output dim: 5, lower bound: -0.0017347, upper bound: 0.0017336

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0062367, 0.0062368
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017584, 0.0017584
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0129736, 0.0129738
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017168, 0.0017169
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0096958, 0.0096957
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026938, 0.0026937
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024451, 0.0024451
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0091248, 0.0091247
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0071018, 0.0071019
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006127, 0.0006127

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0017166, upper bound: 0.0016793
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016774, upper bound: 0.0017175
time: 1.30 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0062368, 0.0062367
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017584, 0.0017584
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0129738, 0.0129736
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017169, 0.0017168
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0096957, 0.0096958
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026937, 0.0026938
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024451, 0.0024451
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0091247, 0.0091248
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0071019, 0.0071018
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006127, 0.0006127

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016894, upper bound: 0.0016893
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016901, upper bound: 0.0016885
time: 1.23 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.24 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 5, lower bound: -0.0017166, upper bound: 0.0016793
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 5, lower bound: -0.0016774, upper bound: 0.0017175
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 5, lower bound: -0.0016894, upper bound: 0.0016893
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.24
Output dim: 5, lower bound: -0.0016901, upper bound: 0.0016885

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061350, 0.0061921
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017297, 0.0017458
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0127621, 0.0128809
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016889, 0.0017046
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0096264, 0.0095376
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026745, 0.0026498
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024276, 0.0024052
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0090595, 0.0089759
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069860, 0.0070510
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006083, 0.0006027

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 62

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016971, upper bound: 0.0016570
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016932, upper bound: 0.0016572
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061911, 0.0061351
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017455, 0.0017297
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0128787, 0.0127623
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017043, 0.0016889
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0095377, 0.0096247
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026499, 0.0026740
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024053, 0.0024272
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0089761, 0.0090579
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0070498, 0.0069861
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006027, 0.0006082

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016766, upper bound: 0.0016712
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016515, upper bound: 0.0017164
time: 1.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060131, 0.0060249
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016953, 0.0016987
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125085, 0.0125331
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016553, 0.0016586
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093665, 0.0093481
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026023, 0.0025972
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023621, 0.0023574
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088149, 0.0087976
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068472, 0.0068606
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005919, 0.0005907

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 165

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016101, upper bound: 0.0016115
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016100, upper bound: 0.0016117
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060250, 0.0060134
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016987, 0.0016954
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125333, 0.0125091
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016586, 0.0016554
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093486, 0.0093666
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025973, 0.0026023
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023576, 0.0023621
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087980, 0.0088150
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068607, 0.0068475
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005908, 0.0005919

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016235, upper bound: 0.0016225
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016235, upper bound: 0.0016225
time: 1.38 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.46 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 5, lower bound: -0.0016971, upper bound: 0.0016570
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 5, lower bound: -0.0016932, upper bound: 0.0016572
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 5, lower bound: -0.0016766, upper bound: 0.0016712
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 5, lower bound: -0.0016515, upper bound: 0.0017164
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 5, lower bound: -0.0016101, upper bound: 0.0016115
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 5, lower bound: -0.0016100, upper bound: 0.0016117
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 5, lower bound: -0.0016235, upper bound: 0.0016225
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.46
Output dim: 5, lower bound: -0.0016235, upper bound: 0.0016225

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060638, 0.0061306
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017096, 0.0017284
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0126140, 0.0127529
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016693, 0.0016876
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0095307, 0.0094269
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026479, 0.0026191
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024035, 0.0023773
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0089695, 0.0088718
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069049, 0.0069810
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006023, 0.0005957

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016656, upper bound: 0.0016171
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016563, upper bound: 0.0016251
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060692, 0.0061209
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017111, 0.0017257
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0126252, 0.0127328
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016707, 0.0016850
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0095157, 0.0094353
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026437, 0.0026214
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023997, 0.0023794
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0089553, 0.0088796
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069110, 0.0069700
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006013, 0.0005963

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016830, upper bound: 0.0016374
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016741, upper bound: 0.0016479
time: 1.86 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061566, 0.0061707
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017358, 0.0017397
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0128070, 0.0128362
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016948, 0.0016987
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0095930, 0.0095711
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026652, 0.0026591
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0024192, 0.0024137
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0090281, 0.0090075
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0070106, 0.0070266
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006062, 0.0006048

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013948, upper bound: 0.0013985
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013948, upper bound: 0.0013985
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0062311, 0.0061006
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017568, 0.0017200
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0129620, 0.0126905
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017153, 0.0016794
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0094841, 0.0096870
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026350, 0.0026913
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023918, 0.0024429
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0089256, 0.0091166
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0070954, 0.0069468
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005993, 0.0006122

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015983, upper bound: 0.0016640
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015983, upper bound: 0.0016639
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059959, 0.0060131
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016905, 0.0016953
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0124727, 0.0125085
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016506, 0.0016553
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093481, 0.0093213
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025972, 0.0025897
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023574, 0.0023507
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087976, 0.0087724
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068275, 0.0068472
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005907, 0.0005890

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016093, upper bound: 0.0016035
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016028, upper bound: 0.0016110
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060013, 0.0060249
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016920, 0.0016987
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0124839, 0.0125331
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016520, 0.0016586
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093665, 0.0093297
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026023, 0.0025921
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023621, 0.0023528
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088149, 0.0087802
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068337, 0.0068606
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005919, 0.0005896

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015912, upper bound: 0.0015916
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015888, upper bound: 0.0015927
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059908, 0.0059712
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016890, 0.0016835
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0124620, 0.0124213
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016491, 0.0016438
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092829, 0.0093133
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025791, 0.0025875
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023410, 0.0023487
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087362, 0.0087649
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068217, 0.0067994
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005866, 0.0005885

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014952, upper bound: 0.0014945
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014952, upper bound: 0.0014945
time: 1.16 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059828, 0.0060134
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016868, 0.0016954
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0124454, 0.0125091
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016470, 0.0016554
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093486, 0.0093009
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025973, 0.0025841
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023576, 0.0023456
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087980, 0.0087532
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068126, 0.0068475
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005908, 0.0005878

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016163, upper bound: 0.0016164
time: 1.73 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016176, upper bound: 0.0016157
time: 1.19 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.64 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0016656, upper bound: 0.0016171
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0016563, upper bound: 0.0016251
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0016830, upper bound: 0.0016374
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0016741, upper bound: 0.0016479
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0013948, upper bound: 0.0013985
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0013948, upper bound: 0.0013985
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0015983, upper bound: 0.0016640
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0015983, upper bound: 0.0016639
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0016093, upper bound: 0.0016035
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0016028, upper bound: 0.0016110
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0015912, upper bound: 0.0015916
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0015888, upper bound: 0.0015927
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0014952, upper bound: 0.0014945
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0014952, upper bound: 0.0014945
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0016163, upper bound: 0.0016164
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.64
Output dim: 5, lower bound: -0.0016176, upper bound: 0.0016157

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060223, 0.0061165
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016979, 0.0017245
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125276, 0.0127236
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016578, 0.0016838
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0095088, 0.0093623
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026418, 0.0026011
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023980, 0.0023610
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0089489, 0.0088110
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068576, 0.0069649
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0006009, 0.0005916

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015536, upper bound: 0.0015158
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015536, upper bound: 0.0015158
time: 1.71 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060458, 0.0060891
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017045, 0.0017167
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125765, 0.0126665
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016643, 0.0016762
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0094662, 0.0093989
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026300, 0.0026113
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023872, 0.0023703
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0089087, 0.0088454
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068844, 0.0069337
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005982, 0.0005940

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015873, upper bound: 0.0015538
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015854, upper bound: 0.0015544
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060026, 0.0060746
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016923, 0.0017126
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0124865, 0.0126363
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016524, 0.0016722
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0094436, 0.0093317
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026237, 0.0025926
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023815, 0.0023533
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088875, 0.0087821
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068351, 0.0069171
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005968, 0.0005897

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016620, upper bound: 0.0016140
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016590, upper bound: 0.0016168
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060244, 0.0060543
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016985, 0.0017069
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125320, 0.0125941
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016584, 0.0016666
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0094121, 0.0093657
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026150, 0.0026021
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023736, 0.0023619
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088578, 0.0088141
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068601, 0.0068941
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005948, 0.0005919

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016554, upper bound: 0.0016220
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016456, upper bound: 0.0016260
time: 2.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061823, 0.0060614
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017430, 0.0017089
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0128605, 0.0126090
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017019, 0.0016686
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0094232, 0.0096112
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026180, 0.0026703
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023764, 0.0024238
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088682, 0.0090452
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0070399, 0.0069022
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005955, 0.0006074

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015524, upper bound: 0.0016205
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015539, upper bound: 0.0016189
time: 1.76 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061919, 0.0061006
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017457, 0.0017200
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0128805, 0.0126905
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0017045, 0.0016794
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0094841, 0.0096260
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026350, 0.0026744
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023918, 0.0024275
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0089256, 0.0090592
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0070508, 0.0069468
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005993, 0.0006083

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015160, upper bound: 0.0015787
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015160, upper bound: 0.0015789
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059566, 0.0059861
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016794, 0.0016877
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123909, 0.0124522
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016397, 0.0016479
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093060, 0.0092602
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025855, 0.0025727
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023468, 0.0023353
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087580, 0.0087148
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067828, 0.0068164
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005881, 0.0005852

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016005, upper bound: 0.0015693
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015644, upper bound: 0.0015941
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059692, 0.0059738
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016829, 0.0016842
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0124171, 0.0124267
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016432, 0.0016445
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092869, 0.0092797
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025802, 0.0025782
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023420, 0.0023402
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087400, 0.0087333
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067971, 0.0068024
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005869, 0.0005864

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015272, upper bound: 0.0015338
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015272, upper bound: 0.0015340
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059314, 0.0059747
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016723, 0.0016845
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123385, 0.0124286
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016328, 0.0016447
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092884, 0.0092210
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025806, 0.0025619
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023424, 0.0023254
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087414, 0.0086780
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067541, 0.0068035
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005870, 0.0005827

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014934, upper bound: 0.0014959
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014934, upper bound: 0.0014961
time: 1.20 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059473, 0.0059551
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016768, 0.0016790
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123716, 0.0123878
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016372, 0.0016393
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092579, 0.0092458
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025721, 0.0025688
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023347, 0.0023317
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087127, 0.0087013
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067723, 0.0067811
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005850, 0.0005843

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 199

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0011523, upper bound: 0.0011533
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0011523, upper bound: 0.0011533
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058599, 0.0059004
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016521, 0.0016635
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121897, 0.0122741
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016131, 0.0016243
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091729, 0.0091098
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025485, 0.0025310
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023133, 0.0022974
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086327, 0.0085734
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066727, 0.0067188
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005797, 0.0005757

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015663, upper bound: 0.0015745
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015741, upper bound: 0.0015662
time: 1.64 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058696, 0.0058921
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016549, 0.0016612
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0122100, 0.0122567
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016158, 0.0016220
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091599, 0.0091250
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025449, 0.0025352
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023100, 0.0023012
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086205, 0.0085876
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066838, 0.0067093
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005788, 0.0005766

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015549, upper bound: 0.0015530
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015543, upper bound: 0.0015535
time: 1.52 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.47 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015536, upper bound: 0.0015158
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015536, upper bound: 0.0015158
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015873, upper bound: 0.0015538
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015854, upper bound: 0.0015544
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0016620, upper bound: 0.0016140
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0016590, upper bound: 0.0016168
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0016554, upper bound: 0.0016220
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0016456, upper bound: 0.0016260
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015524, upper bound: 0.0016205
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015539, upper bound: 0.0016189
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015160, upper bound: 0.0015787
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015160, upper bound: 0.0015789
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0016005, upper bound: 0.0015693
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015644, upper bound: 0.0015941
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015272, upper bound: 0.0015338
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015272, upper bound: 0.0015340
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0014934, upper bound: 0.0014959
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0014934, upper bound: 0.0014961
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0011523, upper bound: 0.0011533
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0011523, upper bound: 0.0011533
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015663, upper bound: 0.0015745
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015741, upper bound: 0.0015662
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015549, upper bound: 0.0015530
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.47
Output dim: 5, lower bound: -0.0015543, upper bound: 0.0015535

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060014, 0.0060628
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016920, 0.0017093
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0124842, 0.0126118
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016521, 0.0016690
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0094252, 0.0093299
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026186, 0.0025921
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023769, 0.0023529
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088702, 0.0087805
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068339, 0.0069037
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005956, 0.0005896

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 54

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 114

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014960, upper bound: 0.0014652
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014960, upper bound: 0.0014652
time: 1.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060458, 0.0060447
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017045, 0.0017042
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125765, 0.0125742
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016643, 0.0016640
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093972, 0.0093989
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026108, 0.0026113
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023698, 0.0023703
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088438, 0.0088454
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068844, 0.0068831
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005938, 0.0005940

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015561, upper bound: 0.0015212
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015558, upper bound: 0.0015218
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059432, 0.0060331
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016756, 0.0017010
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123631, 0.0125500
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016361, 0.0016608
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093791, 0.0092394
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026058, 0.0025670
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023653, 0.0023300
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088268, 0.0086953
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067676, 0.0068699
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005927, 0.0005839

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015080, upper bound: 0.0014626
time: 1.42 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015080, upper bound: 0.0014626
time: 1.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059640, 0.0060152
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016815, 0.0016959
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0124064, 0.0125129
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016418, 0.0016559
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093514, 0.0092718
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025981, 0.0025760
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023583, 0.0023382
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088007, 0.0087258
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067913, 0.0068496
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005909, 0.0005859

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016285, upper bound: 0.0015814
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016252, upper bound: 0.0015867
time: 2.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059277, 0.0059784
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016712, 0.0016855
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123308, 0.0124363
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016318, 0.0016457
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092941, 0.0092153
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025822, 0.0025603
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023438, 0.0023240
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087468, 0.0086726
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067499, 0.0068076
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005873, 0.0005823

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0010798, upper bound: 0.0010822
time: 0.95 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0010798, upper bound: 0.0010822
time: 0.95 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059438, 0.0059576
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016758, 0.0016797
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123643, 0.0123929
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016362, 0.0016400
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092617, 0.0092403
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025732, 0.0025672
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023357, 0.0023303
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087163, 0.0086962
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067683, 0.0067839
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005853, 0.0005839

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016195, upper bound: 0.0015981
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016141, upper bound: 0.0015996
time: 1.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061214, 0.0059965
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017258, 0.0016906
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0127337, 0.0124740
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016851, 0.0016507
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093223, 0.0095164
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025900, 0.0026439
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023510, 0.0023999
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087733, 0.0089560
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069704, 0.0068283
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005891, 0.0006014

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015222, upper bound: 0.0015863
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015163, upper bound: 0.0015890
time: 2.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061175, 0.0059999
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017247, 0.0016916
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0127256, 0.0124810
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016840, 0.0016517
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0093275, 0.0095103
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025915, 0.0026422
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023523, 0.0023984
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087782, 0.0089503
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069660, 0.0068321
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005894, 0.0006010

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015107, upper bound: 0.0015731
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015099, upper bound: 0.0015752
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061548, 0.0060559
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017353, 0.0017074
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0128033, 0.0125976
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016943, 0.0016671
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0094147, 0.0095684
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026157, 0.0026584
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023742, 0.0024130
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0088602, 0.0090049
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0070085, 0.0068959
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005949, 0.0006047

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014593, upper bound: 0.0015139
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014593, upper bound: 0.0015139
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0061481, 0.0061006
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017334, 0.0017200
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0127893, 0.0126905
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016925, 0.0016794
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0094841, 0.0095580
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0026350, 0.0026555
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023918, 0.0024104
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0089256, 0.0089951
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0070009, 0.0069468
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005993, 0.0006040

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014286, upper bound: 0.0014770
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014286, upper bound: 0.0014770
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058556, 0.0059432
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016509, 0.0016756
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121808, 0.0123631
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016119, 0.0016361
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092394, 0.0091031
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025670, 0.0025291
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023301, 0.0022957
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086953, 0.0085671
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066678, 0.0067676
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005839, 0.0005753

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 21

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 199

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014388, upper bound: 0.0014143
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014388, upper bound: 0.0014143
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059119, 0.0058851
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016668, 0.0016592
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0122980, 0.0122421
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016274, 0.0016200
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091490, 0.0091907
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025419, 0.0025535
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023072, 0.0023178
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086102, 0.0086495
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067319, 0.0067014
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005782, 0.0005808

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015256, upper bound: 0.0015550
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015252, upper bound: 0.0015554
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0054759, 0.0054962
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015438, 0.0015496
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0113909, 0.0114333
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015074, 0.0015130
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0085445, 0.0085128
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0023739, 0.0023651
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021548, 0.0021468
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0080413, 0.0080115
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0062354, 0.0062586
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005400, 0.0005380

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015436, upper bound: 0.0015514
time: 1.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015417, upper bound: 0.0015526
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0054580, 0.0055180
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015388, 0.0015557
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0113537, 0.0114786
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015025, 0.0015190
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0085784, 0.0084851
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0023833, 0.0023574
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021634, 0.0021398
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0080732, 0.0079854
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0062150, 0.0062834
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005421, 0.0005362

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 71
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 114

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 71

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015595, upper bound: 0.0015132
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015203, upper bound: 0.0015517
time: 1.79 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.25 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0014960, upper bound: 0.0014652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0014960, upper bound: 0.0014652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015561, upper bound: 0.0015212
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015558, upper bound: 0.0015218
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015080, upper bound: 0.0014626
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015080, upper bound: 0.0014626
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0016285, upper bound: 0.0015814
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0016252, upper bound: 0.0015867
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0010798, upper bound: 0.0010822
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0010798, upper bound: 0.0010822
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0016195, upper bound: 0.0015981
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0016141, upper bound: 0.0015996
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015222, upper bound: 0.0015863
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015163, upper bound: 0.0015890
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015107, upper bound: 0.0015731
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015099, upper bound: 0.0015752
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0014593, upper bound: 0.0015139
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0014593, upper bound: 0.0015139
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0014286, upper bound: 0.0014770
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0014286, upper bound: 0.0014770
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0014388, upper bound: 0.0014143
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0014388, upper bound: 0.0014143
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015256, upper bound: 0.0015550
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015252, upper bound: 0.0015554
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015436, upper bound: 0.0015514
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015417, upper bound: 0.0015526
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015595, upper bound: 0.0015132
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.25
Output dim: 5, lower bound: -0.0015203, upper bound: 0.0015517

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059247, 0.0059773
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016704, 0.0016852
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123246, 0.0124339
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016310, 0.0016454
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092923, 0.0092107
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025817, 0.0025590
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023434, 0.0023228
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087451, 0.0086683
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067465, 0.0068064
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005872, 0.0005821

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 245

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 24

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015834, upper bound: 0.0015414
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015856, upper bound: 0.0015351
time: 2.10 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059272, 0.0059759
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016711, 0.0016848
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123298, 0.0124311
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016316, 0.0016451
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092902, 0.0092145
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025811, 0.0025601
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023429, 0.0023238
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087431, 0.0086719
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067493, 0.0068048
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005871, 0.0005823

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016104, upper bound: 0.0015716
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016106, upper bound: 0.0015712
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059141, 0.0059300
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016674, 0.0016719
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123025, 0.0123357
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016280, 0.0016324
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092189, 0.0091942
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025613, 0.0025544
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023249, 0.0023186
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086760, 0.0086527
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067344, 0.0067526
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005826, 0.0005810

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015686, upper bound: 0.0015554
time: 1.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015778, upper bound: 0.0015482
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059438, 0.0059278
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016758, 0.0016713
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0123643, 0.0123311
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016362, 0.0016318
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092155, 0.0092403
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025603, 0.0025672
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023240, 0.0023303
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086728, 0.0086962
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067683, 0.0067501
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005824, 0.0005839

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0016132, upper bound: 0.0015724
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015489, upper bound: 0.0015990
time: 1.53 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060846, 0.0059612
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017155, 0.0016807
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0126572, 0.0124005
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016750, 0.0016410
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092673, 0.0094592
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025747, 0.0026281
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023371, 0.0023855
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087216, 0.0089022
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069286, 0.0067880
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005856, 0.0005978

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 198

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014876, upper bound: 0.0015487
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014876, upper bound: 0.0015490
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060843, 0.0059598
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017154, 0.0016803
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0126565, 0.0123976
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016749, 0.0016406
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092652, 0.0094587
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025741, 0.0026279
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023365, 0.0023853
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087196, 0.0089017
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069282, 0.0067865
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005855, 0.0005977

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 21

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015029, upper bound: 0.0015726
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014993, upper bound: 0.0015760
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060492, 0.0059424
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017055, 0.0016754
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125835, 0.0123615
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016652, 0.0016358
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092382, 0.0094042
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025666, 0.0026128
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023297, 0.0023716
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086942, 0.0088504
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068882, 0.0067667
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005838, 0.0005943

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014956, upper bound: 0.0015580
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014957, upper bound: 0.0015575
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060614, 0.0059316
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017089, 0.0016723
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0126089, 0.0123389
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016686, 0.0016329
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092213, 0.0094231
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025620, 0.0026180
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023255, 0.0023764
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086783, 0.0088682
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0069021, 0.0067543
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005827, 0.0005955

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 240

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014663, upper bound: 0.0015270
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014663, upper bound: 0.0015270
time: 1.93 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 6.36 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0015834, upper bound: 0.0015414
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0015856, upper bound: 0.0015351
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0016104, upper bound: 0.0015716
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0016106, upper bound: 0.0015712
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0015686, upper bound: 0.0015554
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0015778, upper bound: 0.0015482
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0016132, upper bound: 0.0015724
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0015489, upper bound: 0.0015990
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0014876, upper bound: 0.0015487
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0014876, upper bound: 0.0015490
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0015029, upper bound: 0.0015726
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0014993, upper bound: 0.0015760
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0014956, upper bound: 0.0015580
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0014957, upper bound: 0.0015575
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0014663, upper bound: 0.0015270
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 6.36
Output dim: 5, lower bound: -0.0014663, upper bound: 0.0015270

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0056819, 0.0057513
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016019, 0.0016215
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0118194, 0.0119639
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015641, 0.0015832
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0089411, 0.0088331
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024841, 0.0024541
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022548, 0.0022276
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0084145, 0.0083129
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0064700, 0.0065490
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005650, 0.0005582

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 36

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0011731, upper bound: 0.0011719
time: 0.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0011731, upper bound: 0.0011719
time: 0.98 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0056988, 0.0057378
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016067, 0.0016177
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0118546, 0.0119359
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015688, 0.0015795
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0089201, 0.0088594
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0024783, 0.0024614
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022495, 0.0022342
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0083948, 0.0083376
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0064892, 0.0065337
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005637, 0.0005599

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 37

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014200, upper bound: 0.0013719
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014200, upper bound: 0.0013718
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058740, 0.0059185
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016561, 0.0016686
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0122190, 0.0123117
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016170, 0.0016293
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092010, 0.0091318
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025563, 0.0025371
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023204, 0.0023029
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086592, 0.0085940
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066887, 0.0067394
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005814, 0.0005771

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015806, upper bound: 0.0015313
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015749, upper bound: 0.0015400
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058708, 0.0059227
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016552, 0.0016698
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0122126, 0.0123204
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016161, 0.0016304
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092075, 0.0091269
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025581, 0.0025357
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023220, 0.0023017
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086653, 0.0085894
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066852, 0.0067442
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005819, 0.0005768

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 20

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015919, upper bound: 0.0015460
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015805, upper bound: 0.0015515
time: 2.14 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0055166, 0.0055503
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015553, 0.0015648
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0114757, 0.0115458
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015186, 0.0015279
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0086286, 0.0085762
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0023973, 0.0023827
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0021760, 0.0021628
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0081205, 0.0080712
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0062818, 0.0063202
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005453, 0.0005420

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015482, upper bound: 0.0015150
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015441, upper bound: 0.0015196
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059025, 0.0059612
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016641, 0.0016807
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0122783, 0.0124006
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016248, 0.0016410
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092674, 0.0091761
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025748, 0.0025494
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023371, 0.0023141
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0087217, 0.0086357
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067212, 0.0067881
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005856, 0.0005799

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 245

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015580, upper bound: 0.0015166
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015580, upper bound: 0.0015166
time: 2.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0059690, 0.0058876
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016829, 0.0016599
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0124167, 0.0122474
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016432, 0.0016207
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091529, 0.0092795
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025430, 0.0025781
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023082, 0.0023401
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086139, 0.0087330
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0067969, 0.0067042
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005784, 0.0005864

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 21
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 37

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015047, upper bound: 0.0015547
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015055, upper bound: 0.0015515
time: 2.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060307, 0.0059254
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017003, 0.0016706
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125451, 0.0123260
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016601, 0.0016311
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0092117, 0.0093754
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025593, 0.0026048
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023231, 0.0023643
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086692, 0.0088233
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068672, 0.0067473
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005821, 0.0005925

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 62
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0012845, upper bound: 0.0013128
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0012845, upper bound: 0.0013128
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0060506, 0.0059062
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0017059, 0.0016652
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0125865, 0.0122862
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016656, 0.0016259
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091819, 0.0094064
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025510, 0.0026134
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023155, 0.0023721
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086412, 0.0088524
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0068899, 0.0067255
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005802, 0.0005944

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 62

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 54

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014845, upper bound: 0.0015611
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014845, upper bound: 0.0015610
time: 2.04 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.84 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0011731, upper bound: 0.0011719
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0011731, upper bound: 0.0011719
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0014200, upper bound: 0.0013719
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0014200, upper bound: 0.0013718
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0015806, upper bound: 0.0015313
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0015749, upper bound: 0.0015400
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0015919, upper bound: 0.0015460
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0015805, upper bound: 0.0015515
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0015482, upper bound: 0.0015150
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0015441, upper bound: 0.0015196
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0015580, upper bound: 0.0015166
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0015580, upper bound: 0.0015166
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0015047, upper bound: 0.0015547
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0015055, upper bound: 0.0015515
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0012845, upper bound: 0.0013128
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0012845, upper bound: 0.0013128
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0014845, upper bound: 0.0015611
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.84
Output dim: 5, lower bound: -0.0014845, upper bound: 0.0015610

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058326, 0.0059018
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016444, 0.0016639
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121329, 0.0122770
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016056, 0.0016247
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091751, 0.0090674
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025491, 0.0025192
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023138, 0.0022867
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0086348, 0.0085334
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066416, 0.0067205
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005798, 0.0005730

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 240

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0012098, upper bound: 0.0011895
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0012098, upper bound: 0.0011895
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0058603, 0.0058771
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016522, 0.0016570
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0121906, 0.0122256
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0016132, 0.0016179
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091366, 0.0091105
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025384, 0.0025312
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0023041, 0.0022975
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085986, 0.0085740
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0066732, 0.0066923
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005774, 0.0005757

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 94

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 20

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015213, upper bound: 0.0014975
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015357, upper bound: 0.0014893
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0057768, 0.0058492
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016287, 0.0016491
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0120169, 0.0121675
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015902, 0.0016102
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090932, 0.0089807
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025264, 0.0024951
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022932, 0.0022648
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085577, 0.0084518
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0065781, 0.0066605
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005746, 0.0005675

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015912, upper bound: 0.0015001
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015525, upper bound: 0.0015450
time: 2.09 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0057958, 0.0058286
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016341, 0.0016433
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0120564, 0.0121248
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015955, 0.0016045
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090613, 0.0090102
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025175, 0.0025033
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022851, 0.0022723
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085277, 0.0084796
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0065997, 0.0066371
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005726, 0.0005694

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 24

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 94

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015425, upper bound: 0.0015121
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015421, upper bound: 0.0015145
time: 1.60 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 4.39 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 5, lower bound: -0.0012098, upper bound: 0.0011895
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 5, lower bound: -0.0012098, upper bound: 0.0011895
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 5, lower bound: -0.0015213, upper bound: 0.0014975
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 5, lower bound: -0.0015357, upper bound: 0.0014893
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 4.39
Output dim: 5, lower bound: -0.0015912, upper bound: 0.0015001
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 5, lower bound: -0.0015525, upper bound: 0.0015450
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 5, lower bound: -0.0015425, upper bound: 0.0015121
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.39
Output dim: 5, lower bound: -0.0015421, upper bound: 0.0015145

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0057288, 0.0058658
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0016152, 0.0016538
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0119172, 0.0122021
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015770, 0.0016148
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0091191, 0.0089061
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025336, 0.0024744
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022997, 0.0022460
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0085821, 0.0083817
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0065235, 0.0066795
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005763, 0.0005628

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 165
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 205

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015766, upper bound: 0.0014794
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015348, upper bound: 0.0014859
time: 2.37 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 5.36 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 10, time: 5.36
Output dim: 5, lower bound: -0.0015766, upper bound: 0.0014794
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 10, time: 5.36
Output dim: 5, lower bound: -0.0015348, upper bound: 0.0014859

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0105875, -0.0033245, -0.0105875, -0.0033245, -0.0056573, 0.0058082
1: -0.0059237, -0.0038760, -0.0059237, -0.0038760, -0.0015950, 0.0016376
2: -0.0051464, 0.0099622, -0.0051464, 0.0099622, -0.0117683, 0.0120823
3: 0.0009463, 0.0029456, 0.0009463, 0.0029456, -0.0015573, 0.0015989
4: -0.0013532, 0.0099380, -0.0013532, 0.0099380, -0.0090296, 0.0087949
5: 0.9951303, 0.9982673, 0.9951303, 0.9982673, -0.0025087, 0.0024435
6: 0.0034634, 0.0063109, 0.0034634, 0.0063109, -0.0022771, 0.0022179
7: -0.0104567, 0.0001696, -0.0104567, 0.0001696, -0.0084978, 0.0082770
8: -0.0093249, -0.0010544, -0.0093249, -0.0010544, -0.0064420, 0.0066139
9: -0.0039188, -0.0032052, -0.0039188, -0.0032052, -0.0005706, 0.0005558

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 36
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 198
type: DSZ, layer: 1, pos: 37
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 54
type: DSZ, layer: 1, pos: 245
type: DSZ, layer: 1, pos: 240
type: DSZ, layer: 1, pos: 199
type: DSZ, layer: 1, pos: 20
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 114
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 24
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 94
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 165

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 36

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015458, upper bound: 0.0014506
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0015429, upper bound: 0.0014508
time: 2.28 seconds

## Summary of splitting (split count: 10)
- Time for DS candidates: 4.68 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 11, time: 4.68
Output dim: 5, lower bound: -0.0015458, upper bound: 0.0014506
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 11, time: 4.68
Output dim: 5, lower bound: -0.0015429, upper bound: 0.0014508

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.53 + 279.99 = 283.52 seconds
