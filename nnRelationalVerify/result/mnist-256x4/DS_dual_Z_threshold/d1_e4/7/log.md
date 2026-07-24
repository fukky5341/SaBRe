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
execution time: IAR + RelationalAnalysis = 1.86 + 2.99 = 4.86 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0015904, upper bound: 0.0015903

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 1

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015466, upper bound: 0.0015465
time: 2.40 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015466, upper bound: 0.0015617
time: 2.01 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 4.61 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 4.61
Output dim: 5, lower bound: -0.0015466, upper bound: 0.0015465
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 4.61
Output dim: 5, lower bound: -0.0015466, upper bound: 0.0015617

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063466, 0.0063796
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017893, 0.0017986
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0132021, 0.0132709
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017471, 0.0017562
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0099178, 0.0098665
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027555, 0.0027412
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0025011, 0.0024882
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0093338, 0.0092854
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0072269, 0.0072645
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006267, 0.0006235

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015508, upper bound: 0.0015355
time: 1.86 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015355, upper bound: 0.0015358
time: 2.03 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063796, 0.0063466
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017986, 0.0017893
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0132709, 0.0132021
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017562, 0.0017471
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0098665, 0.0099178
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027412, 0.0027555
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024882, 0.0025011
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092854, 0.0093338
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0072645, 0.0072269
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006235, 0.0006267

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015358, upper bound: 0.0015508
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015355, upper bound: 0.0015508
time: 1.72 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.68 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.68
Output dim: 5, lower bound: -0.0015508, upper bound: 0.0015355
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.68
Output dim: 5, lower bound: -0.0015355, upper bound: 0.0015358
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.68
Output dim: 5, lower bound: -0.0015358, upper bound: 0.0015508
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.68
Output dim: 5, lower bound: -0.0015355, upper bound: 0.0015508

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063172, 0.0063494
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017810, 0.0017901
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0131410, 0.0132081
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017390, 0.0017479
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0098709, 0.0098208
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027424, 0.0027285
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024893, 0.0024767
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092896, 0.0092425
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071934, 0.0072301
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006238, 0.0006206

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015125, upper bound: 0.0014786
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014933, upper bound: 0.0014987
time: 1.99 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063164, 0.0063525
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017808, 0.0017910
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0131394, 0.0132145
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017388, 0.0017487
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0098757, 0.0098195
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027438, 0.0027282
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024905, 0.0024763
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092941, 0.0092413
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0071925, 0.0072336
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006241, 0.0006205

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014787, upper bound: 0.0014789
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014929, upper bound: 0.0014987
time: 1.94 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063525, 0.0063164
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017910, 0.0017808
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0132145, 0.0131394
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017487, 0.0017388
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0098195, 0.0098757
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027282, 0.0027438
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024763, 0.0024905
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092413, 0.0092941
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0072336, 0.0071925
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006205, 0.0006241

Time for backsubstitution: 1.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014987, upper bound: 0.0014929
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014789, upper bound: 0.0015125
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0063494, 0.0063172
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017901, 0.0017810
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0132081, 0.0131410
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0017479, 0.0017390
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0098208, 0.0098709
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0027285, 0.0027424
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024767, 0.0024893
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0092425, 0.0092896
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0072301, 0.0071934
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006206, 0.0006238

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 13
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 13

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014986, upper bound: 0.0014933
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014786, upper bound: 0.0015125
time: 1.88 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 5.07 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 5, lower bound: -0.0015125, upper bound: 0.0014786
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 5, lower bound: -0.0014933, upper bound: 0.0014987
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 5.07
Output dim: 5, lower bound: -0.0014787, upper bound: 0.0014789
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 5, lower bound: -0.0014929, upper bound: 0.0014987
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 5, lower bound: -0.0014987, upper bound: 0.0014929
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 5, lower bound: -0.0014789, upper bound: 0.0015125
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 5, lower bound: -0.0014986, upper bound: 0.0014933
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 5.07
Output dim: 5, lower bound: -0.0014786, upper bound: 0.0015125

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0060807, 0.0061624
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017144, 0.0017374
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0126490, 0.0128190
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016739, 0.0016964
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0095801, 0.0094531
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026616, 0.0026264
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024160, 0.0023839
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0090160, 0.0088964
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0069241, 0.0070171
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006054, 0.0005974

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0015022, upper bound: 0.0014270
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014675
time: 2.05 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061336, 0.0061129
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017293, 0.0017235
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0127591, 0.0127161
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016885, 0.0016828
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0095032, 0.0095353
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026403, 0.0026492
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023966, 0.0024047
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0089436, 0.0089738
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0069843, 0.0069608
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006005, 0.0006026

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014823, upper bound: 0.0014427
time: 1.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014394, upper bound: 0.0014879
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061297, 0.0061160
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017282, 0.0017243
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0127510, 0.0127225
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016874, 0.0016836
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0095080, 0.0095293
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026416, 0.0026475
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023978, 0.0024032
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0089481, 0.0089682
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0069799, 0.0069643
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006008, 0.0006022

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014427
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014880
time: 2.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061160, 0.0061297
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017243, 0.0017282
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0127225, 0.0127510
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016836, 0.0016874
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0095293, 0.0095080
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026475, 0.0026416
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024032, 0.0023978
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0089682, 0.0089481
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0069643, 0.0069799
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006022, 0.0006008

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014393
time: 2.17 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014818
time: 2.07 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061664, 0.0060799
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017386, 0.0017141
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0128275, 0.0126474
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016975, 0.0016737
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0094519, 0.0095864
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026260, 0.0026634
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023836, 0.0024176
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0088952, 0.0090219
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070218, 0.0069232
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005973, 0.0006058

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014523
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0015021
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061129, 0.0061336
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017235, 0.0017293
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0127161, 0.0127591
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016828, 0.0016885
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0095353, 0.0095032
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026492, 0.0026403
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0024047, 0.0023966
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0089738, 0.0089436
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0069608, 0.0069843
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0006026, 0.0006005

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014879, upper bound: 0.0014394
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014823
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0061624, 0.0060807
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0017374, 0.0017144
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0128190, 0.0126490
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016964, 0.0016739
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0094531, 0.0095801
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0026264, 0.0026616
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023839, 0.0024160
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0088964, 0.0090160
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0070171, 0.0069241
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005974, 0.0006054

Time for backsubstitution: 1.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014523
time: 2.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0015021
time: 1.75 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0015022, upper bound: 0.0014270
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014675
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014823, upper bound: 0.0014427
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014394, upper bound: 0.0014879
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014427
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014880
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014393
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014818
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014523
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0015021
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014879, upper bound: 0.0014394
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014823
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0014523
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.20
Output dim: 5, lower bound: -0.0014271, upper bound: 0.0015021

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0058963, 0.0060052
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016624, 0.0016931
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0122655, 0.0124921
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016231, 0.0016531
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0093358, 0.0091665
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025938, 0.0025467
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023543, 0.0023116
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0087860, 0.0086267
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0067141, 0.0068382
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005900, 0.0005793

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014025
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0014751, upper bound: 0.0013918
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0059492, 0.0059580
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016773, 0.0016798
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0123755, 0.0123939
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016377, 0.0016401
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0092624, 0.0092487
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025734, 0.0025696
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023358, 0.0023324
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0087170, 0.0087041
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0067744, 0.0067844
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005853, 0.0005845

Time for backsubstitution: 1.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014186
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014077
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0059793, 0.0059285
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016858, 0.0016715
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0124382, 0.0123326
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016460, 0.0016320
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0092166, 0.0092956
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025606, 0.0025826
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023243, 0.0023442
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0086738, 0.0087482
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0068087, 0.0067509
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005824, 0.0005874

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014632
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014544
time: 2.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0059762, 0.0059316
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016849, 0.0016723
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0124318, 0.0123389
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016451, 0.0016329
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0092213, 0.0092907
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025620, 0.0025812
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023255, 0.0023430
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0086783, 0.0087436
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0068052, 0.0067543
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005827, 0.0005871

Time for backsubstitution: 1.32 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014635
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014544
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0059607, 0.0059453
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016805, 0.0016762
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0123995, 0.0123675
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016409, 0.0016366
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0092427, 0.0092666
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025679, 0.0025745
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023309, 0.0023369
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0086984, 0.0087209
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0067875, 0.0067700
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005841, 0.0005856

Time for backsubstitution: 1.35 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014558
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014493
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0060077, 0.0058955
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016938, 0.0016622
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0124972, 0.0122638
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016538, 0.0016229
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0091652, 0.0093396
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025464, 0.0025948
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023113, 0.0023553
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0086255, 0.0087896
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0068410, 0.0067132
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005792, 0.0005902

Time for backsubstitution: 1.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014751
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014696
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0059285, 0.0059793
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016715, 0.0016858
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0123326, 0.0124382
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016320, 0.0016460
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0092956, 0.0092166
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025826, 0.0025606
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023442, 0.0023243
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0087482, 0.0086738
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0067509, 0.0068087
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005874, 0.0005824

Time for backsubstitution: 1.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014143
time: 2.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014050
time: 2.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0059580, 0.0059492
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016798, 0.0016773
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0123939, 0.0123755
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016401, 0.0016377
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0092487, 0.0092624
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025696, 0.0025734
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023324, 0.0023358
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0087041, 0.0087170
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0067844, 0.0067744
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005845, 0.0005853

Time for backsubstitution: 1.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014563
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014493
time: 1.96 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0092659, -0.0009024, -0.0092659, -0.0009024, -0.0060052, 0.0058963
1: -0.0055511, -0.0031931, -0.0055511, -0.0031931, -0.0016931, 0.0016624
2: -0.0023972, 0.0150006, -0.0023972, 0.0150006, -0.0124921, 0.0122655
3: 0.0013101, 0.0036124, 0.0013101, 0.0036124, -0.0016531, 0.0016231
4: -0.0051186, 0.0078834, -0.0051186, 0.0078834, -0.0091665, 0.0093358
5: 0.9940842, 0.9976964, 0.9940842, 0.9976964, -0.0025467, 0.0025938
6: 0.0025138, 0.0057927, 0.0025138, 0.0057927, -0.0023116, 0.0023543
7: -0.0140003, -0.0017640, -0.0140003, -0.0017640, -0.0086267, 0.0087860
8: -0.0078199, 0.0017036, -0.0078199, 0.0017036, -0.0068382, 0.0067141
9: -0.0041567, -0.0033351, -0.0041567, -0.0033351, -0.0005793, 0.0005900

Time for backsubstitution: 1.43 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 42
type: DSZ, layer: 1, pos: 76
type: DSZ, layer: 1, pos: 90
type: DSZ, layer: 1, pos: 107
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 144
type: DSZ, layer: 1, pos: 159
type: DSZ, layer: 1, pos: 161
type: DSZ, layer: 1, pos: 176
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 187
type: DSZ, layer: 1, pos: 204
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 235

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 42

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014752
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014696
time: 2.08 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.56 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014025
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0014751, upper bound: 0.0013918
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014186
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014077
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014632
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014544
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014635
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014544
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014558
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014493
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014751
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014696
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014143
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014050
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014563
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014493
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014752
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 5.56
Output dim: 5, lower bound: -0.0013919, upper bound: 0.0014696

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 4.86 + 121.06 = 125.92 seconds
