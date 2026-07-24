## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 7)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0014790720000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0065673, 0.0065673)
1: (-0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018516, 0.0018516)
2: (-0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0136613, 0.0136613)
3: (0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0018079, 0.0018079)
4: (-0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0102096, 0.0102096)
5: (0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028365, 0.0028365)
6: (0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025747, 0.0025747)
7: (-0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0096084, 0.0096084)
8: (-0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0074782, 0.0074782)
9: (-0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006452, 0.0006452)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.96 + 2.66 = 3.62 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0015904, upper bound: 0.0015903

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015796, upper bound: 0.0015795
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015795, upper bound: 0.0015796
time: 1.83 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.62 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.62
Output dim: 5, lower bound: -0.0015796, upper bound: 0.0015795
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.62
Output dim: 5, lower bound: -0.0015795, upper bound: 0.0015796

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0065346, 0.0065338
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018423, 0.0018421
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0135933, 0.0135916
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017989, 0.0017986
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0101575, 0.0101587
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028221, 0.0028224
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025616, 0.0025619
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0095593, 0.0095605
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0074410, 0.0074401
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006419, 0.0006420

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015787, upper bound: 0.0015463
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015463, upper bound: 0.0015786
time: 1.47 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0065338, 0.0065346
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018421, 0.0018423
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0135916, 0.0135933
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017986, 0.0017989
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0101587, 0.0101575
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028224, 0.0028221
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025619, 0.0025616
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0095605, 0.0095593
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0074401, 0.0074410
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006420, 0.0006419

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015658, upper bound: 0.0015753
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015752, upper bound: 0.0015658
time: 1.75 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.52 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 5, lower bound: -0.0015787, upper bound: 0.0015463
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 5, lower bound: -0.0015463, upper bound: 0.0015786
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 5, lower bound: -0.0015658, upper bound: 0.0015753
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.52
Output dim: 5, lower bound: -0.0015752, upper bound: 0.0015658

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064992, 0.0065059
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018324, 0.0018343
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0135196, 0.0135337
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017891, 0.0017910
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0101142, 0.0101037
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028100, 0.0028071
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025507, 0.0025480
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0095186, 0.0095087
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0074007, 0.0074084
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006392, 0.0006385

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015559, upper bound: 0.0015218
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015544, upper bound: 0.0015236
time: 1.74 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0065068, 0.0064984
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018345, 0.0018321
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0135356, 0.0135180
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017912, 0.0017889
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0101025, 0.0101156
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028068, 0.0028104
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025477, 0.0025510
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0095076, 0.0095199
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0074094, 0.0073997
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006384, 0.0006392

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014487, upper bound: 0.0014780
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014487, upper bound: 0.0014780
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0065205, 0.0065198
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018384, 0.0018382
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0135640, 0.0135626
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017950, 0.0017948
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0101358, 0.0101369
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028160, 0.0028163
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025561, 0.0025564
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0095389, 0.0095399
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0074249, 0.0074242
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006405, 0.0006406

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015536, upper bound: 0.0015671
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015536, upper bound: 0.0015644
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0065190, 0.0065213
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018380, 0.0018386
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0135609, 0.0135656
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017946, 0.0017952
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0101381, 0.0101346
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028167, 0.0028157
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025567, 0.0025558
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0095411, 0.0095378
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0074233, 0.0074258
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006407, 0.0006404

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015743, upper bound: 0.0015379
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015382, upper bound: 0.0015649
time: 2.10 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 4.38 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 5, lower bound: -0.0015559, upper bound: 0.0015218
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 5, lower bound: -0.0015544, upper bound: 0.0015236
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 4.38
Output dim: 5, lower bound: -0.0014487, upper bound: 0.0014780
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 4.38
Output dim: 5, lower bound: -0.0014487, upper bound: 0.0014780
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 5, lower bound: -0.0015536, upper bound: 0.0015671
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 5, lower bound: -0.0015536, upper bound: 0.0015644
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 5, lower bound: -0.0015743, upper bound: 0.0015379
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 4.38
Output dim: 5, lower bound: -0.0015382, upper bound: 0.0015649

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064528, 0.0064698
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018193, 0.0018241
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0134231, 0.0134584
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017763, 0.0017810
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100580, 0.0100316
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027944, 0.0027871
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025365, 0.0025298
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0094657, 0.0094408
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073478, 0.0073671
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006356, 0.0006339

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015290, upper bound: 0.0014952
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015208, upper bound: 0.0014970
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064619, 0.0064595
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018218, 0.0018212
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0134420, 0.0134371
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017788, 0.0017782
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100421, 0.0100457
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027900, 0.0027910
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025325, 0.0025334
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0094507, 0.0094541
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073582, 0.0073555
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006346, 0.0006348

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015130, upper bound: 0.0015190
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015502, upper bound: 0.0015154
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064989, 0.0064948
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018323, 0.0018311
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0135191, 0.0135105
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017890, 0.0017879
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100969, 0.0101033
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028052, 0.0028070
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025463, 0.0025479
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0095023, 0.0095084
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0074004, 0.0073957
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006381, 0.0006385

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015216, upper bound: 0.0015413
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015216, upper bound: 0.0015339
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064955, 0.0064983
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018313, 0.0018321
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0135119, 0.0135178
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017881, 0.0017889
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0101023, 0.0100980
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028067, 0.0028055
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025477, 0.0025466
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0095074, 0.0095033
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073964, 0.0073996
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006384, 0.0006381

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014983, upper bound: 0.0015049
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014983, upper bound: 0.0015535
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064843, 0.0064943
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018282, 0.0018310
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0134886, 0.0135094
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017850, 0.0017877
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100961, 0.0100806
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028050, 0.0028007
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025461, 0.0025422
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0095015, 0.0094869
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073837, 0.0073950
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006380, 0.0006370

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015634, upper bound: 0.0014959
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015150, upper bound: 0.0015258
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064919, 0.0064865
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018303, 0.0018288
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0135044, 0.0134933
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017871, 0.0017856
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100841, 0.0100923
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0028017, 0.0028039
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025431, 0.0025451
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0094902, 0.0094980
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073923, 0.0073863
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006373, 0.0006378

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015189, upper bound: 0.0015325
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015155, upper bound: 0.0015424
time: 1.78 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0015290, upper bound: 0.0014952
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0015208, upper bound: 0.0014970
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0015130, upper bound: 0.0015190
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0015502, upper bound: 0.0015154
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0015216, upper bound: 0.0015413
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0015216, upper bound: 0.0015339
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0014983, upper bound: 0.0015049
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0014983, upper bound: 0.0015535
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0015634, upper bound: 0.0014959
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0015150, upper bound: 0.0015258
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0015189, upper bound: 0.0015325
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.45
Output dim: 5, lower bound: -0.0015155, upper bound: 0.0015424

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064404, 0.0064608
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018158, 0.0018215
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0133972, 0.0134398
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017729, 0.0017785
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100440, 0.0100123
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027905, 0.0027817
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025330, 0.0025249
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0094526, 0.0094227
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073337, 0.0073569
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006347, 0.0006327

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014746, upper bound: 0.0014746
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014737, upper bound: 0.0014739
time: 1.87 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064437, 0.0064573
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018167, 0.0018206
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0134043, 0.0134326
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017738, 0.0017776
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100387, 0.0100175
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027890, 0.0027832
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025316, 0.0025263
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0094475, 0.0094276
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073375, 0.0073530
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006344, 0.0006330

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014565, upper bound: 0.0014652
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014566, upper bound: 0.0014947
time: 2.21 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064488, 0.0064451
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018182, 0.0018171
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0134148, 0.0134072
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017752, 0.0017742
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100197, 0.0100254
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027838, 0.0027853
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025268, 0.0025283
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0094297, 0.0094350
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073433, 0.0073391
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006332, 0.0006335

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014876, upper bound: 0.0014916
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014876, upper bound: 0.0014936
time: 2.07 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064475, 0.0064466
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018178, 0.0018175
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0134121, 0.0134103
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017749, 0.0017746
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100220, 0.0100234
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027844, 0.0027848
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025274, 0.0025277
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0094318, 0.0094331
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073418, 0.0073408
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006333, 0.0006334

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014991, upper bound: 0.0015002
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014991, upper bound: 0.0015018
time: 2.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063165, 0.0063004
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017809, 0.0017763
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0131396, 0.0131060
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017388, 0.0017344
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097946, 0.0098197
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027212, 0.0027282
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024701, 0.0024764
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092178, 0.0092414
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071926, 0.0071743
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006190, 0.0006205

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014979, upper bound: 0.0015165
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014969, upper bound: 0.0015174
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063045, 0.0063120
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017775, 0.0017796
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0131146, 0.0131302
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017355, 0.0017376
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0098127, 0.0098010
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027263, 0.0027230
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024746, 0.0024717
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092348, 0.0092239
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071790, 0.0071875
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006201, 0.0006194

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014882, upper bound: 0.0014984
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014882, upper bound: 0.0015107
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063164, 0.0063493
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017808, 0.0017901
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0131393, 0.0132079
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017388, 0.0017478
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0098707, 0.0098195
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027424, 0.0027282
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024893, 0.0024763
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092895, 0.0092413
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071925, 0.0072300
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006238, 0.0006205

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014647, upper bound: 0.0014797
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014647, upper bound: 0.0014704
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063459, 0.0063192
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017891, 0.0017816
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0132007, 0.0131452
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017469, 0.0017396
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0098239, 0.0098654
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027294, 0.0027409
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024774, 0.0024879
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092454, 0.0092844
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0072261, 0.0071957
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006208, 0.0006234

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014482, upper bound: 0.0015087
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014590, upper bound: 0.0015040
time: 1.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063041, 0.0063440
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017774, 0.0017886
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0131138, 0.0131968
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017354, 0.0017464
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0098624, 0.0098005
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027401, 0.0027229
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024872, 0.0024715
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092817, 0.0092233
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071785, 0.0072239
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006232, 0.0006193

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015401, upper bound: 0.0014857
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014763, upper bound: 0.0014758
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063336, 0.0063141
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017857, 0.0017802
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0131751, 0.0131346
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017435, 0.0017382
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0098160, 0.0098462
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027272, 0.0027356
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024754, 0.0024831
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092379, 0.0092664
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0072121, 0.0071899
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006203, 0.0006222

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014835, upper bound: 0.0015177
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014835, upper bound: 0.0015110
time: 1.71 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064795, 0.0064775
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018268, 0.0018263
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0134786, 0.0134745
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017837, 0.0017831
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100700, 0.0100731
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027978, 0.0027986
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025395, 0.0025403
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0094770, 0.0094799
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073782, 0.0073760
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006364, 0.0006366

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0012354, upper bound: 0.0012424
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0012354, upper bound: 0.0012424
time: 1.21 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064830, 0.0064742
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018278, 0.0018253
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0134859, 0.0134676
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017846, 0.0017822
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100649, 0.0100785
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027963, 0.0028001
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025382, 0.0025417
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0094721, 0.0094850
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073822, 0.0073722
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006360, 0.0006369

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014513, upper bound: 0.0014839
time: 2.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014513, upper bound: 0.0015052
time: 2.01 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.91 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014746, upper bound: 0.0014746
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014737, upper bound: 0.0014739
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014565, upper bound: 0.0014652
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014566, upper bound: 0.0014947
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014876, upper bound: 0.0014916
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014876, upper bound: 0.0014936
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014991, upper bound: 0.0015002
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014991, upper bound: 0.0015018
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014979, upper bound: 0.0015165
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014969, upper bound: 0.0015174
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014882, upper bound: 0.0014984
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014882, upper bound: 0.0015107
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014647, upper bound: 0.0014797
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014647, upper bound: 0.0014704
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014482, upper bound: 0.0015087
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014590, upper bound: 0.0015040
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0015401, upper bound: 0.0014857
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014763, upper bound: 0.0014758
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014835, upper bound: 0.0015177
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014835, upper bound: 0.0015110
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0012354, upper bound: 0.0012424
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0012354, upper bound: 0.0012424
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014513, upper bound: 0.0014839
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.91
Output dim: 5, lower bound: -0.0014513, upper bound: 0.0015052

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062046, 0.0061457
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017493, 0.0017327
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0129068, 0.0127843
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017080, 0.0016918
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0095542, 0.0096458
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026544, 0.0026799
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024094, 0.0024325
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0089916, 0.0090777
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070652, 0.0069982
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006038, 0.0006096

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014359, upper bound: 0.0014739
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014359, upper bound: 0.0014736
time: 1.64 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064365, 0.0064363
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018147, 0.0018146
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0133891, 0.0133888
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017718, 0.0017718
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100060, 0.0100062
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027800, 0.0027800
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025234, 0.0025234
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0094167, 0.0094170
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073292, 0.0073290
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006323, 0.0006323

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014252, upper bound: 0.0014331
time: 2.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014247, upper bound: 0.0014573
time: 2.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0064399, 0.0064328
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018156, 0.0018136
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0133962, 0.0133816
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017728, 0.0017708
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0100005, 0.0100115
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027784, 0.0027815
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025220, 0.0025248
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0094116, 0.0094220
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0073331, 0.0073251
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006320, 0.0006327

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014949, upper bound: 0.0014487
time: 2.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014409, upper bound: 0.0014809
time: 2.28 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063855, 0.0063959
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018003, 0.0018033
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0132831, 0.0133048
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017578, 0.0017607
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0099432, 0.0099270
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027625, 0.0027580
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025075, 0.0025034
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0093577, 0.0093424
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0072712, 0.0072831
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006283, 0.0006273

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0012066, upper bound: 0.0011908
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0012066, upper bound: 0.0011908
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063968, 0.0063861
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018035, 0.0018005
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0133067, 0.0132844
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017609, 0.0017580
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0099279, 0.0099446
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027583, 0.0027629
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025037, 0.0025079
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0093433, 0.0093589
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0072841, 0.0072719
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006274, 0.0006284

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 76

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015028, upper bound: 0.0014743
time: 2.19 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014723, upper bound: 0.0014747
time: 2.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062710, 0.0062640
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017680, 0.0017661
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130450, 0.0130305
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017263, 0.0017244
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097381, 0.0097491
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027055, 0.0027086
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024558, 0.0024586
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091647, 0.0091749
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071409, 0.0071329
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006154, 0.0006161

Time for backsubstitution: 1.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014757, upper bound: 0.0015068
time: 2.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014737, upper bound: 0.0014919
time: 2.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062793, 0.0062549
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017704, 0.0017635
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130621, 0.0130115
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017286, 0.0017219
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097240, 0.0097618
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027016, 0.0027121
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024522, 0.0024618
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091514, 0.0091870
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071502, 0.0071225
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006145, 0.0006169

Time for backsubstitution: 1.05 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014619
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014584
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062914, 0.0063017
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017738, 0.0017767
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130875, 0.0131087
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017319, 0.0017347
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097966, 0.0097808
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027218, 0.0027174
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024706, 0.0024666
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092197, 0.0092048
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071641, 0.0071757
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006191, 0.0006181

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014878, upper bound: 0.0014984
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014878, upper bound: 0.0014977
time: 2.11 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062950, 0.0062989
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017748, 0.0017759
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130950, 0.0131031
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017329, 0.0017340
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097924, 0.0097864
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027206, 0.0027189
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024695, 0.0024680
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092157, 0.0092100
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071682, 0.0071726
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006188, 0.0006184

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014878, upper bound: 0.0015106
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014878, upper bound: 0.0015088
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061231, 0.0061448
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017263, 0.0017325
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0127374, 0.0127825
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016856, 0.0016916
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0095528, 0.0095191
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026541, 0.0026447
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024091, 0.0024006
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0089903, 0.0089586
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0069725, 0.0069972
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006037, 0.0006016

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014438, upper bound: 0.0014589
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014438, upper bound: 0.0014589
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062787, 0.0062503
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017702, 0.0017622
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130609, 0.0130018
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017284, 0.0017206
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097167, 0.0097609
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026996, 0.0027119
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024504, 0.0024616
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091445, 0.0091861
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071495, 0.0071172
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006140, 0.0006168

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014139, upper bound: 0.0014577
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014139, upper bound: 0.0015063
time: 2.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062769, 0.0062520
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017697, 0.0017627
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130573, 0.0130054
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017279, 0.0017211
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097194, 0.0097582
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027003, 0.0027111
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024511, 0.0024609
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091470, 0.0091836
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071476, 0.0071192
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006142, 0.0006167

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014581, upper bound: 0.0014666
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014310, upper bound: 0.0015031
time: 1.77 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062270, 0.0062579
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017556, 0.0017643
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0129535, 0.0130177
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017142, 0.0017227
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097286, 0.0096806
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027029, 0.0026896
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024534, 0.0024413
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091557, 0.0091106
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070908, 0.0071259
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006148, 0.0006118

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015116, upper bound: 0.0014423
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014329, upper bound: 0.0014582
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063152, 0.0062913
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017805, 0.0017738
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0131370, 0.0130872
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017385, 0.0017319
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097806, 0.0098178
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027173, 0.0027277
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024665, 0.0024759
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092046, 0.0092396
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071912, 0.0071640
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006181, 0.0006204

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014620, upper bound: 0.0014978
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014620, upper bound: 0.0014975
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063108, 0.0062945
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017792, 0.0017746
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0131277, 0.0130937
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017372, 0.0017327
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097854, 0.0098109
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027187, 0.0027257
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024677, 0.0024742
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092092, 0.0092331
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071861, 0.0071675
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006184, 0.0006200

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014673, upper bound: 0.0014923
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014673, upper bound: 0.0014943
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062680, 0.0063119
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017672, 0.0017795
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130388, 0.0131300
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017255, 0.0017375
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0098125, 0.0097444
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027262, 0.0027073
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024746, 0.0024574
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092347, 0.0091705
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071374, 0.0071874
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006201, 0.0006158

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014573, upper bound: 0.0014558
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014568, upper bound: 0.0014586
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063178, 0.0062592
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017812, 0.0017647
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0131422, 0.0130205
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017392, 0.0017231
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097307, 0.0098217
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027035, 0.0027288
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024539, 0.0024769
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091577, 0.0092433
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071941, 0.0071274
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006149, 0.0006207

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014112, upper bound: 0.0014484
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014112, upper bound: 0.0014943
time: 1.91 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.99 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014359, upper bound: 0.0014739
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014359, upper bound: 0.0014736
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014252, upper bound: 0.0014331
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014247, upper bound: 0.0014573
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014949, upper bound: 0.0014487
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014409, upper bound: 0.0014809
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0012066, upper bound: 0.0011908
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0012066, upper bound: 0.0011908
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0015028, upper bound: 0.0014743
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014723, upper bound: 0.0014747
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014757, upper bound: 0.0015068
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014737, upper bound: 0.0014919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014619
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014392, upper bound: 0.0014584
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014878, upper bound: 0.0014984
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014878, upper bound: 0.0014977
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014878, upper bound: 0.0015106
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014878, upper bound: 0.0015088
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014438, upper bound: 0.0014589
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014438, upper bound: 0.0014589
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014139, upper bound: 0.0014577
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014139, upper bound: 0.0015063
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014581, upper bound: 0.0014666
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014310, upper bound: 0.0015031
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0015116, upper bound: 0.0014423
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014329, upper bound: 0.0014582
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014620, upper bound: 0.0014978
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014620, upper bound: 0.0014975
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014673, upper bound: 0.0014923
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014673, upper bound: 0.0014943
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014573, upper bound: 0.0014558
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014568, upper bound: 0.0014586
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014112, upper bound: 0.0014484
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.99
Output dim: 5, lower bound: -0.0014112, upper bound: 0.0014943

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062600, 0.0062830
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017649, 0.0017714
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130220, 0.0130699
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017233, 0.0017296
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097676, 0.0097318
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027137, 0.0027038
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024633, 0.0024542
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091924, 0.0091587
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071283, 0.0071545
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006173, 0.0006150

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014330, upper bound: 0.0014392
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014854, upper bound: 0.0014293
time: 1.96 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062897, 0.0062529
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017733, 0.0017629
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130839, 0.0130073
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017314, 0.0017213
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097208, 0.0097781
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027007, 0.0027166
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024515, 0.0024659
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091484, 0.0092022
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071621, 0.0071202
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006143, 0.0006179

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014286, upper bound: 0.0014645
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014286, upper bound: 0.0014650
time: 2.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063845, 0.0063773
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0018000, 0.0017980
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0132810, 0.0132661
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017575, 0.0017556
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0099143, 0.0099254
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027545, 0.0027576
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025002, 0.0025030
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0093304, 0.0093409
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0072700, 0.0072619
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006265, 0.0006272

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014662, upper bound: 0.0014489
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014380, upper bound: 0.0014404
time: 2.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062023, 0.0061850
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017487, 0.0017438
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0129020, 0.0128661
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017074, 0.0017026
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0096153, 0.0096421
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026714, 0.0026789
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024248, 0.0024316
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0090491, 0.0090743
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070626, 0.0070429
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006076, 0.0006093

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 235

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014530, upper bound: 0.0014845
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014530, upper bound: 0.0014845
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061920, 0.0061933
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017458, 0.0017461
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0128806, 0.0128834
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017045, 0.0017049
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0096282, 0.0096262
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026750, 0.0026744
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024281, 0.0024276
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0090612, 0.0090593
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070509, 0.0070524
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006084, 0.0006083

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014721, upper bound: 0.0014760
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014543, upper bound: 0.0014780
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062883, 0.0062982
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017729, 0.0017757
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130810, 0.0131014
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017311, 0.0017338
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097912, 0.0097759
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027203, 0.0027160
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024692, 0.0024653
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092146, 0.0092002
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071605, 0.0071717
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006187, 0.0006178

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014443, upper bound: 0.0014520
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014343, upper bound: 0.0014435
time: 2.17 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062879, 0.0062991
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017728, 0.0017759
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130802, 0.0131034
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017310, 0.0017340
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097926, 0.0097753
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027207, 0.0027159
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024696, 0.0024652
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092160, 0.0091997
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071601, 0.0071728
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006188, 0.0006177

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014770, upper bound: 0.0014704
time: 2.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014618, upper bound: 0.0014729
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062919, 0.0062954
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017739, 0.0017749
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130885, 0.0130958
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017320, 0.0017330
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097870, 0.0097815
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027191, 0.0027176
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024681, 0.0024668
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092106, 0.0092055
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071646, 0.0071686
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006185, 0.0006181

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014637, upper bound: 0.0014818
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014617, upper bound: 0.0014830
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062915, 0.0062963
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017738, 0.0017752
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130877, 0.0130977
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017319, 0.0017333
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097884, 0.0097809
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027195, 0.0027174
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024685, 0.0024666
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092120, 0.0092049
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071642, 0.0071697
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006186, 0.0006181

Time for backsubstitution: 1.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 107

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014331, upper bound: 0.0014564
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014331, upper bound: 0.0015063
time: 1.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0060458, 0.0059420
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017045, 0.0016753
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0125766, 0.0123605
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016643, 0.0016357
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0092375, 0.0093990
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025664, 0.0026113
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023296, 0.0023703
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0086935, 0.0088455
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0068844, 0.0067662
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005838, 0.0005940

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 161

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013881, upper bound: 0.0014637
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013747, upper bound: 0.0014769
time: 2.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062521, 0.0062187
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017627, 0.0017533
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130056, 0.0129362
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017211, 0.0017119
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0096677, 0.0097196
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026860, 0.0027004
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024381, 0.0024511
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0090984, 0.0091472
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071193, 0.0070813
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006109, 0.0006142

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014013, upper bound: 0.0014597
time: 2.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013881, upper bound: 0.0014745
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0059896, 0.0060550
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016887, 0.0017071
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0124596, 0.0125956
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016488, 0.0016668
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0094132, 0.0093115
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026153, 0.0025870
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023739, 0.0023482
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0088588, 0.0087632
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0068204, 0.0068949
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005949, 0.0005884

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013838, upper bound: 0.0014002
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013838, upper bound: 0.0013894
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062600, 0.0062388
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017649, 0.0017589
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130221, 0.0129779
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017233, 0.0017174
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0096989, 0.0097319
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026946, 0.0027038
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024459, 0.0024543
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091277, 0.0091588
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071283, 0.0071041
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006129, 0.0006150

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014302, upper bound: 0.0014723
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014302, upper bound: 0.0014652
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062627, 0.0062364
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017657, 0.0017583
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130277, 0.0129729
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017240, 0.0017168
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0096952, 0.0097361
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026936, 0.0027050
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024450, 0.0024553
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091242, 0.0091627
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071314, 0.0071014
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006127, 0.0006153

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014552, upper bound: 0.0014684
time: 1.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014323, upper bound: 0.0014701
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062471, 0.0062431
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017613, 0.0017602
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0129952, 0.0129870
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017197, 0.0017186
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097057, 0.0097118
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026965, 0.0026982
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024476, 0.0024492
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091341, 0.0091399
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071136, 0.0071091
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006133, 0.0006137

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 1

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014067, upper bound: 0.0014398
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014365, upper bound: 0.0014352
time: 1.91 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062595, 0.0062352
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017648, 0.0017579
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130210, 0.0129705
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017231, 0.0017164
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0096933, 0.0097311
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026931, 0.0027036
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024445, 0.0024540
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091225, 0.0091580
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071277, 0.0071001
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006126, 0.0006149

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 76

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014674, upper bound: 0.0014943
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014635, upper bound: 0.0014932
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061752, 0.0060853
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017410, 0.0017157
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0128458, 0.0126587
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016999, 0.0016752
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0094604, 0.0096001
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026284, 0.0026672
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023858, 0.0024210
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0089032, 0.0090348
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070318, 0.0069294
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005978, 0.0006067

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 187

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0013956, upper bound: 0.0014839
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014042, upper bound: 0.0014724
time: 2.19 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 5.00 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014330, upper bound: 0.0014392
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014854, upper bound: 0.0014293
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014286, upper bound: 0.0014645
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014286, upper bound: 0.0014650
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014662, upper bound: 0.0014489
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014380, upper bound: 0.0014404
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014530, upper bound: 0.0014845
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014530, upper bound: 0.0014845
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014721, upper bound: 0.0014760
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014543, upper bound: 0.0014780
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014443, upper bound: 0.0014520
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014343, upper bound: 0.0014435
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014770, upper bound: 0.0014704
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014618, upper bound: 0.0014729
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014637, upper bound: 0.0014818
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014617, upper bound: 0.0014830
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014331, upper bound: 0.0014564
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014331, upper bound: 0.0015063
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0013881, upper bound: 0.0014637
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0013747, upper bound: 0.0014769
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014013, upper bound: 0.0014597
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0013881, upper bound: 0.0014745
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0013838, upper bound: 0.0014002
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0013838, upper bound: 0.0013894
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014302, upper bound: 0.0014723
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014302, upper bound: 0.0014652
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014552, upper bound: 0.0014684
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014323, upper bound: 0.0014701
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014067, upper bound: 0.0014398
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014365, upper bound: 0.0014352
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014674, upper bound: 0.0014943
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014635, upper bound: 0.0014932
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0013956, upper bound: 0.0014839
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 5.00
Output dim: 5, lower bound: -0.0014042, upper bound: 0.0014724

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061738, 0.0062054
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017406, 0.0017495
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0128427, 0.0129084
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016995, 0.0017082
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0096470, 0.0095978
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026802, 0.0026666
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024328, 0.0024204
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0090789, 0.0090326
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070301, 0.0070661
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006096, 0.0006065

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 187

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014082, upper bound: 0.0014212
time: 1.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014771, upper bound: 0.0014174
time: 2.16 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061487, 0.0061325
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017336, 0.0017290
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0127906, 0.0127568
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016926, 0.0016882
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0095336, 0.0095589
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026487, 0.0026558
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024042, 0.0024106
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0089722, 0.0089960
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070016, 0.0069831
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006025, 0.0006041

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 204

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 144

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014307, upper bound: 0.0014618
time: 2.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014270, upper bound: 0.0014632
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061497, 0.0061306
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017338, 0.0017285
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0127927, 0.0127530
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016929, 0.0016877
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0095308, 0.0095605
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026479, 0.0026562
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024035, 0.0024110
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0089695, 0.0089975
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070027, 0.0069810
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006023, 0.0006042

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 159

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0010982, upper bound: 0.0011066
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0010982, upper bound: 0.0011066
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062460, 0.0062563
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017610, 0.0017639
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0129929, 0.0130144
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017194, 0.0017222
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097261, 0.0097101
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027022, 0.0026977
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024528, 0.0024487
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091534, 0.0091383
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071123, 0.0071241
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006146, 0.0006136

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014375, upper bound: 0.0014557
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014375, upper bound: 0.0014560
time: 2.13 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062568, 0.0062495
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017640, 0.0017620
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130153, 0.0130002
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017224, 0.0017204
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0097156, 0.0097269
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026993, 0.0027024
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024501, 0.0024530
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091434, 0.0091541
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071246, 0.0071163
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006140, 0.0006147

Time for backsubstitution: 1.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 176

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013967, upper bound: 0.0014127
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013999, upper bound: 0.0014110
time: 2.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0060281, 0.0059588
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016995, 0.0016800
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0125396, 0.0123955
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016594, 0.0016403
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0092636, 0.0093713
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025737, 0.0026036
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023361, 0.0023633
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0087181, 0.0088195
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0068642, 0.0067853
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005854, 0.0005922

Time for backsubstitution: 1.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014065, upper bound: 0.0014798
time: 2.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014065, upper bound: 0.0014801
time: 2.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062542, 0.0062294
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017633, 0.0017563
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130100, 0.0129585
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017217, 0.0017148
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0096844, 0.0097229
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026906, 0.0027013
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024423, 0.0024520
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091141, 0.0091503
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071217, 0.0070935
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006120, 0.0006144

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014208, upper bound: 0.0014513
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014208, upper bound: 0.0014664
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0062537, 0.0062298
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017631, 0.0017564
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0130090, 0.0129593
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017215, 0.0017150
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0096850, 0.0097221
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026908, 0.0027011
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024424, 0.0024518
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0091146, 0.0091496
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071211, 0.0070939
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006120, 0.0006144

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 161

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014449, upper bound: 0.0014833
time: 2.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014802, upper bound: 0.0014690
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0060938, 0.0059936
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017181, 0.0016898
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0126763, 0.0124678
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016775, 0.0016499
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0093177, 0.0094735
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025887, 0.0026320
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023498, 0.0023891
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0087690, 0.0089156
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0069390, 0.0068249
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005888, 0.0005987

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 176

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013659, upper bound: 0.0014579
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013677, upper bound: 0.0014588
time: 1.86 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 4.68 seconds
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014082, upper bound: 0.0014212
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014771, upper bound: 0.0014174
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014307, upper bound: 0.0014618
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014270, upper bound: 0.0014632
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0010982, upper bound: 0.0011066
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0010982, upper bound: 0.0011066
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014375, upper bound: 0.0014557
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014375, upper bound: 0.0014560
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0013967, upper bound: 0.0014127
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0013999, upper bound: 0.0014110
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014065, upper bound: 0.0014798
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014065, upper bound: 0.0014801
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014208, upper bound: 0.0014513
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014208, upper bound: 0.0014664
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014449, upper bound: 0.0014833
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0014802, upper bound: 0.0014690
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0013659, upper bound: 0.0014579
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 4.68
Output dim: 5, lower bound: -0.0013677, upper bound: 0.0014588

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0059477, 0.0058707
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016769, 0.0016552
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0123724, 0.0122123
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016373, 0.0016161
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0091267, 0.0092463
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025357, 0.0025689
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023016, 0.0023318
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0085893, 0.0087018
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0067727, 0.0066851
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005768, 0.0005843

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 90

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013543, upper bound: 0.0014310
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013620, upper bound: 0.0014265
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0059457, 0.0058784
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016763, 0.0016573
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0123682, 0.0122282
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016367, 0.0016182
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0091386, 0.0092432
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025390, 0.0025680
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023046, 0.0023310
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0086005, 0.0086989
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0067704, 0.0066938
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005775, 0.0005841

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 204

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014146, upper bound: 0.0014446
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014003, upper bound: 0.0014793
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061733, 0.0061409
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017405, 0.0017313
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0128418, 0.0127743
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016994, 0.0016905
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0095467, 0.0095971
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026524, 0.0026664
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024075, 0.0024203
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0089845, 0.0090320
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070296, 0.0069927
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006033, 0.0006065

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 90

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014225, upper bound: 0.0014615
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014170, upper bound: 0.0014629
time: 2.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061648, 0.0061488
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017381, 0.0017336
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0128240, 0.0127908
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016970, 0.0016927
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0095591, 0.0095838
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026558, 0.0026627
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024107, 0.0024169
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0089961, 0.0090195
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070199, 0.0070017
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006041, 0.0006056

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 107

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 159

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014184, upper bound: 0.0014444
time: 2.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014184, upper bound: 0.0014463
time: 1.78 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 4.95 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.95
Output dim: 5, lower bound: -0.0013543, upper bound: 0.0014310
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.95
Output dim: 5, lower bound: -0.0013620, upper bound: 0.0014265
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.95
Output dim: 5, lower bound: -0.0014146, upper bound: 0.0014446
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.95
Output dim: 5, lower bound: -0.0014003, upper bound: 0.0014793
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.95
Output dim: 5, lower bound: -0.0014225, upper bound: 0.0014615
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.95
Output dim: 5, lower bound: -0.0014170, upper bound: 0.0014629
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.95
Output dim: 5, lower bound: -0.0014184, upper bound: 0.0014444
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.95
Output dim: 5, lower bound: -0.0014184, upper bound: 0.0014463

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0059281, 0.0058515
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016714, 0.0016497
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0123317, 0.0121723
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016319, 0.0016108
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0090968, 0.0092160
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025274, 0.0025605
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0022941, 0.0023241
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0085611, 0.0086733
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0067504, 0.0066631
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005749, 0.0005824

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 235
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013609, upper bound: 0.0014156
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013609, upper bound: 0.0014693
time: 2.25 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 5.13 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 5.13
Output dim: 5, lower bound: -0.0013609, upper bound: 0.0014156
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 5.13
Output dim: 5, lower bound: -0.0013609, upper bound: 0.0014693

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.62 + 352.40 = 356.02 seconds
