## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.00390625
execution index: (1, 4, 9)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.000586925


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0033858, 0.0033858)
1: (-0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009546, 0.0009546)
2: (-0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0070432, 0.0070432)
3: (0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009321, 0.0009321)
4: (0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0052637, 0.0052637)
5: (0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014624, 0.0014624)
6: (0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0013274, 0.0013274)
7: (-0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0049537, 0.0049537)
8: (-0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0038555, 0.0038555)
9: (-0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003326, 0.0003326)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 0.97 + 2.08 = 3.04 seconds
status: Status.UNKNOWN
relational distance
Output dim: 5, lower bound: -0.0006905, upper bound: 0.0006905

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006861, upper bound: 0.0006759
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006758, upper bound: 0.0006861
time: 1.13 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.29 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 5, lower bound: -0.0006861, upper bound: 0.0006759
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.29
Output dim: 5, lower bound: -0.0006758, upper bound: 0.0006861

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0033825, 0.0033849
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009537, 0.0009543
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0070364, 0.0070413
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009311, 0.0009318
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0052622, 0.0052585
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014620, 0.0014610
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0013271, 0.0013261
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0049524, 0.0049489
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0038517, 0.0038544
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003325, 0.0003323

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006737, upper bound: 0.0006580
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006685, upper bound: 0.0006634
time: 1.25 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0033849, 0.0033825
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009543, 0.0009537
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0070413, 0.0070364
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009318, 0.0009311
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0052585, 0.0052622
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014610, 0.0014620
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0013261, 0.0013271
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0049489, 0.0049524
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0038544, 0.0038517
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003323, 0.0003325

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006562, upper bound: 0.0006662
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006562, upper bound: 0.0006689
time: 1.33 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.04 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.04
Output dim: 5, lower bound: -0.0006737, upper bound: 0.0006580
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.04
Output dim: 5, lower bound: -0.0006685, upper bound: 0.0006634
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.04
Output dim: 5, lower bound: -0.0006562, upper bound: 0.0006662
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.04
Output dim: 5, lower bound: -0.0006562, upper bound: 0.0006689

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0033214, 0.0033323
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009364, 0.0009395
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0069093, 0.0069320
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009143, 0.0009173
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0051805, 0.0051636
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014393, 0.0014346
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0013065, 0.0013022
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0048754, 0.0048595
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037821, 0.0037946
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003274, 0.0003263

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005336, upper bound: 0.0005282
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005336, upper bound: 0.0005282
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0033297, 0.0033238
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009388, 0.0009371
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0069266, 0.0069142
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009166, 0.0009150
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0051673, 0.0051765
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014356, 0.0014382
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0013031, 0.0013054
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0048630, 0.0048716
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037916, 0.0037849
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003265, 0.0003271

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006432, upper bound: 0.0005998
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006004, upper bound: 0.0006369
time: 1.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0033367, 0.0033435
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009407, 0.0009426
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0069411, 0.0069551
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009185, 0.0009204
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0051978, 0.0051873
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014441, 0.0014412
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0013108, 0.0013082
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0048917, 0.0048818
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037995, 0.0038072
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003285, 0.0003278

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006433, upper bound: 0.0006528
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006433, upper bound: 0.0006531
time: 1.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0033458, 0.0033341
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009433, 0.0009400
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0069600, 0.0069355
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009211, 0.0009178
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0051832, 0.0052015
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014400, 0.0014451
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0013071, 0.0013117
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0048779, 0.0048952
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0038099, 0.0037965
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003275, 0.0003287

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006455, upper bound: 0.0006574
time: 1.70 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006455, upper bound: 0.0006602
time: 1.25 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.82 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.82
Output dim: 5, lower bound: -0.0005336, upper bound: 0.0005282
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.82
Output dim: 5, lower bound: -0.0005336, upper bound: 0.0005282
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 5, lower bound: -0.0006432, upper bound: 0.0005998
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 5, lower bound: -0.0006004, upper bound: 0.0006369
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 5, lower bound: -0.0006433, upper bound: 0.0006528
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 5, lower bound: -0.0006433, upper bound: 0.0006531
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 5, lower bound: -0.0006455, upper bound: 0.0006574
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.82
Output dim: 5, lower bound: -0.0006455, upper bound: 0.0006602

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0027219, 0.0027855
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007674, 0.0007853
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0056620, 0.0057945
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007493, 0.0007668
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0043304, 0.0042315
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0012031, 0.0011756
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010921, 0.0010671
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0040754, 0.0039823
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0030994, 0.0031719
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002737, 0.0002674

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 178

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006376, upper bound: 0.0005898
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006306, upper bound: 0.0005940
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0027885, 0.0027159
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007862, 0.0007657
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0058005, 0.0056497
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007676, 0.0007476
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0042222, 0.0043350
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0011731, 0.0012044
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010648, 0.0010932
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0039736, 0.0040797
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0031752, 0.0030927
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002668, 0.0002739

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 220

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005712, upper bound: 0.0006068
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0006072
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032894, 0.0032959
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009274, 0.0009292
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0068427, 0.0068561
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009055, 0.0009073
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0051238, 0.0051138
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014236, 0.0014208
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012922, 0.0012896
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0048221, 0.0048126
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037457, 0.0037530
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003238, 0.0003232

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006391, upper bound: 0.0006497
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006391, upper bound: 0.0006496
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032891, 0.0032971
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009273, 0.0009296
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0068421, 0.0068587
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009054, 0.0009076
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0051257, 0.0051134
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014241, 0.0014206
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012926, 0.0012895
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0048239, 0.0048122
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037454, 0.0037544
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003239, 0.0003231

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005241, upper bound: 0.0005292
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005241, upper bound: 0.0005292
time: 1.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0033253, 0.0033179
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009375, 0.0009354
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0069173, 0.0069020
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009154, 0.0009134
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0051581, 0.0051696
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014331, 0.0014363
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0013008, 0.0013037
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0048543, 0.0048651
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037866, 0.0037781
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003260, 0.0003267

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006182, upper bound: 0.0006298
time: 1.74 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006200, upper bound: 0.0006305
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0033302, 0.0033135
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009389, 0.0009342
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0069275, 0.0068928
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009167, 0.0009122
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0051512, 0.0051772
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014312, 0.0014384
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012991, 0.0013056
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0048479, 0.0048723
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037921, 0.0037731
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003255, 0.0003272

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006158, upper bound: 0.0006312
time: 1.84 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006158, upper bound: 0.0006349
time: 1.36 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 4.07 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0006376, upper bound: 0.0005898
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0006306, upper bound: 0.0005940
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0005712, upper bound: 0.0006068
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0006072
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0006391, upper bound: 0.0006497
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0006391, upper bound: 0.0006496
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0005241, upper bound: 0.0005292
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0005241, upper bound: 0.0005292
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0006182, upper bound: 0.0006298
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0006200, upper bound: 0.0006305
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0006158, upper bound: 0.0006312
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 4.07
Output dim: 5, lower bound: -0.0006158, upper bound: 0.0006349

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0026838, 0.0027564
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007567, 0.0007771
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0055829, 0.0057339
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007388, 0.0007588
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0042852, 0.0041723
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0011905, 0.0011592
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010807, 0.0010522
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0040328, 0.0039266
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0030561, 0.0031388
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002708, 0.0002637

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006091, upper bound: 0.0005580
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006067, upper bound: 0.0005580
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0026941, 0.0027475
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007596, 0.0007746
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0056042, 0.0057153
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007416, 0.0007563
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0042713, 0.0041883
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0011867, 0.0011636
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010772, 0.0010562
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0040198, 0.0039416
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0030678, 0.0031286
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002699, 0.0002647

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004866, upper bound: 0.0004610
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004866, upper bound: 0.0004610
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0027828, 0.0027140
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007846, 0.0007652
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0057887, 0.0056456
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007660, 0.0007471
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0042192, 0.0043261
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0011722, 0.0012019
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010640, 0.0010910
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0039707, 0.0040714
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0031688, 0.0030904
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002666, 0.0002734

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005555, upper bound: 0.0005891
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005537, upper bound: 0.0005904
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0027885, 0.0027103
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007862, 0.0007641
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0058005, 0.0056379
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007676, 0.0007461
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0042134, 0.0043350
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0011706, 0.0012044
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010626, 0.0010932
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0039653, 0.0040797
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0031752, 0.0030862
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002663, 0.0002739

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 220

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005519, upper bound: 0.0005869
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005497, upper bound: 0.0005886
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032902, 0.0032936
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009276, 0.0009286
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0068443, 0.0068514
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009057, 0.0009067
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0051203, 0.0051150
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014226, 0.0014211
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012913, 0.0012899
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0048188, 0.0048138
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037466, 0.0037504
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003236, 0.0003232

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006257, upper bound: 0.0006344
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006257, upper bound: 0.0006437
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032871, 0.0032943
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009268, 0.0009288
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0068379, 0.0068529
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009049, 0.0009069
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0051214, 0.0051102
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014229, 0.0014198
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012916, 0.0012887
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0048198, 0.0048093
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037431, 0.0037513
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003236, 0.0003229

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006255, upper bound: 0.0006363
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006255, upper bound: 0.0006374
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032840, 0.0032773
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009259, 0.0009240
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0068314, 0.0068174
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009040, 0.0009022
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050949, 0.0051054
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014155, 0.0014184
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012849, 0.0012875
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047949, 0.0048047
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037395, 0.0037319
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003220, 0.0003226

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006138, upper bound: 0.0006267
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006138, upper bound: 0.0006267
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032855, 0.0032766
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009263, 0.0009238
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0068346, 0.0068161
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009045, 0.0009020
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050939, 0.0051078
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014152, 0.0014191
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012846, 0.0012881
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047939, 0.0048070
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037413, 0.0037311
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003219, 0.0003228

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 205

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005948, upper bound: 0.0005960
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005852, upper bound: 0.0006046
time: 1.50 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032450, 0.0032443
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009149, 0.0009147
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0067502, 0.0067488
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008933, 0.0008931
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050436, 0.0050447
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014013, 0.0014016
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012719, 0.0012722
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047466, 0.0047476
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036951, 0.0036943
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003187, 0.0003188

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005967, upper bound: 0.0006123
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005967, upper bound: 0.0006147
time: 2.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032604, 0.0032283
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009192, 0.0009102
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0067823, 0.0067155
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008975, 0.0008887
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050187, 0.0050687
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013944, 0.0014082
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012657, 0.0012782
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047232, 0.0047702
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037126, 0.0036761
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003172, 0.0003203

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006102, upper bound: 0.0006221
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006032, upper bound: 0.0006294
time: 1.35 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 3.72 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0006091, upper bound: 0.0005580
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0006067, upper bound: 0.0005580
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0004866, upper bound: 0.0004610
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0004866, upper bound: 0.0004610
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0005555, upper bound: 0.0005891
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0005537, upper bound: 0.0005904
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0005519, upper bound: 0.0005869
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0005497, upper bound: 0.0005886
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0006257, upper bound: 0.0006344
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0006257, upper bound: 0.0006437
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0006255, upper bound: 0.0006363
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0006255, upper bound: 0.0006374
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0006138, upper bound: 0.0006267
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0006138, upper bound: 0.0006267
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0005948, upper bound: 0.0005960
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0005852, upper bound: 0.0006046
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0005967, upper bound: 0.0006123
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0005967, upper bound: 0.0006147
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0006102, upper bound: 0.0006221
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 3.72
Output dim: 5, lower bound: -0.0006032, upper bound: 0.0006294

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0026492, 0.0027299
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007469, 0.0007697
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0055108, 0.0056787
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007293, 0.0007515
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0042439, 0.0041184
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0011791, 0.0011442
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010703, 0.0010386
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0039940, 0.0038759
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0030166, 0.0031086
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002682, 0.0002603

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004883, upper bound: 0.0004583
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004883, upper bound: 0.0004583
time: 1.04 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0026506, 0.0027218
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007473, 0.0007674
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0055137, 0.0056619
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007297, 0.0007493
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0042313, 0.0041206
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0011756, 0.0011448
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010671, 0.0010392
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0039821, 0.0038780
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0030182, 0.0030993
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002674, 0.0002604

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003912, upper bound: 0.0003756
time: 0.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0003912, upper bound: 0.0003756
time: 0.91 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0026920, 0.0026473
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007590, 0.0007464
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0056000, 0.0055069
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007411, 0.0007288
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0041155, 0.0041851
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0011434, 0.0011627
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010379, 0.0010554
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0038732, 0.0039386
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0030655, 0.0030145
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002601, 0.0002645

Time for backsubstitution: 0.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005402, upper bound: 0.0005724
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005301, upper bound: 0.0005727
time: 1.95 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0027120, 0.0026233
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007646, 0.0007396
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0056416, 0.0054569
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007466, 0.0007221
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0040782, 0.0042162
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0011330, 0.0011714
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010285, 0.0010633
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0038380, 0.0039679
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0030882, 0.0029871
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002577, 0.0002664

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 220
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005503, upper bound: 0.0005829
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005509, upper bound: 0.0005869
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0027468, 0.0026594
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007744, 0.0007498
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0057138, 0.0055322
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007561, 0.0007321
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0041344, 0.0042701
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0011487, 0.0011864
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010426, 0.0010769
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0038909, 0.0040187
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0031277, 0.0030283
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002613, 0.0002698

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005442, upper bound: 0.0005642
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005425, upper bound: 0.0005848
time: 1.38 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032233, 0.0032340
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009088, 0.0009118
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0067051, 0.0067275
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008873, 0.0008903
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050277, 0.0050110
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013968, 0.0013922
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012679, 0.0012637
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047316, 0.0047159
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036704, 0.0036826
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003177, 0.0003167

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006133, upper bound: 0.0006220
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006133, upper bound: 0.0006229
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032321, 0.0032267
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009113, 0.0009097
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0067235, 0.0067123
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008897, 0.0008883
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050163, 0.0050247
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013937, 0.0013960
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012650, 0.0012672
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047209, 0.0047288
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036804, 0.0036743
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003170, 0.0003175

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006098, upper bound: 0.0006298
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006098, upper bound: 0.0006268
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032064, 0.0032391
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009040, 0.0009132
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0066700, 0.0067379
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008827, 0.0008917
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050355, 0.0049847
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013990, 0.0013849
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012699, 0.0012571
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047390, 0.0046912
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036512, 0.0036883
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003182, 0.0003150

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004392, upper bound: 0.0004405
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004392, upper bound: 0.0004405
time: 0.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032300, 0.0032136
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009107, 0.0009060
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0067191, 0.0066850
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008892, 0.0008847
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0049959, 0.0050214
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013880, 0.0013951
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012599, 0.0012663
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047017, 0.0047257
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036780, 0.0036594
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003157, 0.0003173

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006093, upper bound: 0.0006234
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006093, upper bound: 0.0006207
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032787, 0.0032712
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009244, 0.0009223
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0068204, 0.0068048
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009026, 0.0009005
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050855, 0.0050972
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014129, 0.0014161
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012825, 0.0012854
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047860, 0.0047970
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037335, 0.0037249
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003214, 0.0003221

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005513, upper bound: 0.0005608
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005513, upper bound: 0.0005977
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032779, 0.0032709
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009242, 0.0009222
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0068188, 0.0068040
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0009024, 0.0009004
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050849, 0.0050959
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014127, 0.0014158
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012823, 0.0012851
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047855, 0.0047958
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037326, 0.0037245
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003213, 0.0003220

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 205

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005899, upper bound: 0.0005929
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005832, upper bound: 0.0006012
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032508, 0.0032482
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009165, 0.0009158
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0067624, 0.0067570
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008949, 0.0008942
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050498, 0.0050538
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014030, 0.0014041
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012735, 0.0012745
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047524, 0.0047562
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037018, 0.0036988
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003191, 0.0003194

Time for backsubstitution: 0.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 185

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005726, upper bound: 0.0005772
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005664, upper bound: 0.0005804
time: 1.85 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032584, 0.0032419
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009187, 0.0009140
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0067782, 0.0067439
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008970, 0.0008924
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050399, 0.0050656
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014002, 0.0014074
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012710, 0.0012775
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047432, 0.0047673
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037104, 0.0036916
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003185, 0.0003201

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005774, upper bound: 0.0005945
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005756, upper bound: 0.0005945
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0031196, 0.0031217
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008795, 0.0008801
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0064894, 0.0064938
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008588, 0.0008594
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0048531, 0.0048498
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013483, 0.0013474
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012239, 0.0012230
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0045673, 0.0045642
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035523, 0.0035547
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003067, 0.0003065

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005392, upper bound: 0.0005477
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005378, upper bound: 0.0005838
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0031201, 0.0031189
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008797, 0.0008793
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0064904, 0.0064880
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008589, 0.0008586
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0048487, 0.0048505
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013471, 0.0013476
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012228, 0.0012232
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0045632, 0.0045649
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035529, 0.0035515
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003064, 0.0003065

Time for backsubstitution: 0.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005832, upper bound: 0.0006006
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005832, upper bound: 0.0006027
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0031798, 0.0031554
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008965, 0.0008896
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0066147, 0.0065639
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008753, 0.0008686
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0049055, 0.0049434
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013629, 0.0013734
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012371, 0.0012467
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0046166, 0.0046523
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036209, 0.0035931
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003100, 0.0003124

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005897, upper bound: 0.0006078
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005897, upper bound: 0.0006103
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0031866, 0.0031477
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008984, 0.0008875
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0066287, 0.0065479
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008772, 0.0008665
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0048935, 0.0049539
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013596, 0.0013763
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012341, 0.0012493
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0046053, 0.0046621
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036286, 0.0035843
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003092, 0.0003131

Time for backsubstitution: 1.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005414, upper bound: 0.0005614
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005414, upper bound: 0.0006031
time: 1.42 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 4.65 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0004883, upper bound: 0.0004583
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0004883, upper bound: 0.0004583
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0003912, upper bound: 0.0003756
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0003912, upper bound: 0.0003756
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005402, upper bound: 0.0005724
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005301, upper bound: 0.0005727
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005503, upper bound: 0.0005829
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005509, upper bound: 0.0005869
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005442, upper bound: 0.0005642
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005425, upper bound: 0.0005848
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0006133, upper bound: 0.0006220
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0006133, upper bound: 0.0006229
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0006098, upper bound: 0.0006298
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0006098, upper bound: 0.0006268
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0004392, upper bound: 0.0004405
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0004392, upper bound: 0.0004405
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0006093, upper bound: 0.0006234
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0006093, upper bound: 0.0006207
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005513, upper bound: 0.0005608
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005513, upper bound: 0.0005977
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005899, upper bound: 0.0005929
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005832, upper bound: 0.0006012
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005726, upper bound: 0.0005772
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005664, upper bound: 0.0005804
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005774, upper bound: 0.0005945
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005756, upper bound: 0.0005945
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005392, upper bound: 0.0005477
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005378, upper bound: 0.0005838
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005832, upper bound: 0.0006006
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005832, upper bound: 0.0006027
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005897, upper bound: 0.0006078
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005897, upper bound: 0.0006103
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005414, upper bound: 0.0005614
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.65
Output dim: 5, lower bound: -0.0005414, upper bound: 0.0006031

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0031406, 0.0031557
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008855, 0.0008897
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0065331, 0.0065645
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008646, 0.0008687
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0049059, 0.0048824
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013630, 0.0013565
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012372, 0.0012313
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0046170, 0.0045949
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035762, 0.0035934
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003100, 0.0003085

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005952, upper bound: 0.0006038
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005952, upper bound: 0.0006048
time: 1.48 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0031434, 0.0031514
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008862, 0.0008885
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0065390, 0.0065555
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008653, 0.0008675
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0048991, 0.0048868
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013611, 0.0013577
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012355, 0.0012324
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0046106, 0.0045990
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035794, 0.0035885
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003096, 0.0003088

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0006074, upper bound: 0.0006048
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005952, upper bound: 0.0006056
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032103, 0.0031959
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009051, 0.0009010
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0066781, 0.0066480
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008837, 0.0008798
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0049683, 0.0049908
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013803, 0.0013866
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012529, 0.0012586
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0046757, 0.0046969
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036556, 0.0036391
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003140, 0.0003154

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005916, upper bound: 0.0006112
time: 1.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005916, upper bound: 0.0006132
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032013, 0.0031998
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009026, 0.0009022
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0066593, 0.0066563
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008812, 0.0008809
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0049745, 0.0049767
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013821, 0.0013827
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012545, 0.0012551
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0046816, 0.0046836
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036453, 0.0036437
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003144, 0.0003145

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005481, upper bound: 0.0005618
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005481, upper bound: 0.0005989
time: 2.12 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0031909, 0.0031677
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008996, 0.0008931
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0066377, 0.0065895
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008784, 0.0008720
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0049246, 0.0049606
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013682, 0.0013782
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012419, 0.0012510
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0046346, 0.0046685
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036335, 0.0036071
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003112, 0.0003135

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005838, upper bound: 0.0005975
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005838, upper bound: 0.0005975
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0031841, 0.0031733
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008977, 0.0008947
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0066237, 0.0066012
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008765, 0.0008736
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0049333, 0.0049501
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013706, 0.0013753
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012441, 0.0012483
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0046428, 0.0046586
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036258, 0.0036135
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003118, 0.0003128

Time for backsubstitution: 1.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004352, upper bound: 0.0004429
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004352, upper bound: 0.0004429
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0027186, 0.0026472
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007665, 0.0007463
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0056553, 0.0055067
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007484, 0.0007287
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0041154, 0.0042264
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0011434, 0.0011742
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0010378, 0.0010658
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0038730, 0.0039775
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0030957, 0.0030144
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002601, 0.0002671

Time for backsubstitution: 1.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005406, upper bound: 0.0005860
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005406, upper bound: 0.0005863
time: 2.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032431, 0.0032424
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009144, 0.0009141
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0067463, 0.0067448
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008928, 0.0008926
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050406, 0.0050418
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0014004, 0.0014008
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012712, 0.0012715
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047438, 0.0047449
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036929, 0.0036921
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003185, 0.0003186

Time for backsubstitution: 1.06 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005502, upper bound: 0.0005615
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005502, upper bound: 0.0005620
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032501, 0.0032360
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009163, 0.0009124
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0067609, 0.0067316
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008947, 0.0008908
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0050308, 0.0050527
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013977, 0.0014038
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012687, 0.0012742
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0047345, 0.0047551
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0037009, 0.0036849
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003179, 0.0003193

Time for backsubstitution: 1.08 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004392, upper bound: 0.0004413
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004392, upper bound: 0.0004413
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032093, 0.0031944
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009048, 0.0009006
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0066761, 0.0066449
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008835, 0.0008793
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0049660, 0.0049893
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013797, 0.0013862
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012523, 0.0012582
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0046735, 0.0046955
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036545, 0.0036374
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003138, 0.0003153

Time for backsubstitution: 1.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005479, upper bound: 0.0005619
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005446, upper bound: 0.0005626
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0032108, 0.0031950
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0009053, 0.0009008
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0066792, 0.0066463
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008839, 0.0008795
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0049670, 0.0049916
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013800, 0.0013868
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012526, 0.0012588
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0046745, 0.0046977
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0036562, 0.0036382
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003139, 0.0003154

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005593, upper bound: 0.0005770
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005593, upper bound: 0.0005778
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030357, 0.0030541
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008559, 0.0008611
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0063150, 0.0063531
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008357, 0.0008407
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047479, 0.0047194
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013191, 0.0013112
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011973, 0.0011902
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044683, 0.0044415
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034568, 0.0034777
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003000, 0.0002982

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005668, upper bound: 0.0005855
time: 1.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005719, upper bound: 0.0005891
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030616, 0.0030346
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008632, 0.0008556
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0063687, 0.0063125
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008428, 0.0008354
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047176, 0.0047596
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013107, 0.0013224
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011897, 0.0012003
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044398, 0.0044793
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034863, 0.0034555
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002981, 0.0003008

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005236, upper bound: 0.0005377
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005236, upper bound: 0.0005747
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030968, 0.0030929
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008731, 0.0008720
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0064420, 0.0064338
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008525, 0.0008514
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0048082, 0.0048144
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013359, 0.0013376
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012126, 0.0012141
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0045250, 0.0045309
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035264, 0.0035219
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003038, 0.0003042

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005917
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0006041
time: 1.37 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0031187, 0.0030724
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008793, 0.0008662
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0064875, 0.0063913
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008585, 0.0008458
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047765, 0.0048484
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013270, 0.0013470
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012046, 0.0012227
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044952, 0.0045629
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035513, 0.0034986
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003018, 0.0003064

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 170

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005933, upper bound: 0.0005949
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0006065
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0026337, 0.0025203
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007425, 0.0007106
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0054785, 0.0052427
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007250, 0.0006938
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0039181, 0.0040943
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0010886, 0.0011375
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0009881, 0.0010325
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0036873, 0.0038532
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0029990, 0.0028699
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002476, 0.0002587

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005269, upper bound: 0.0005896
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005269, upper bound: 0.0005912
time: 1.41 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.12 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005952, upper bound: 0.0006038
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005952, upper bound: 0.0006048
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0006074, upper bound: 0.0006048
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005952, upper bound: 0.0006056
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005916, upper bound: 0.0006112
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005916, upper bound: 0.0006132
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005481, upper bound: 0.0005618
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005481, upper bound: 0.0005989
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005838, upper bound: 0.0005975
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005838, upper bound: 0.0005975
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0004352, upper bound: 0.0004429
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0004352, upper bound: 0.0004429
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005406, upper bound: 0.0005860
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005406, upper bound: 0.0005863
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005502, upper bound: 0.0005615
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005502, upper bound: 0.0005620
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0004392, upper bound: 0.0004413
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0004392, upper bound: 0.0004413
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005479, upper bound: 0.0005619
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005446, upper bound: 0.0005626
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005593, upper bound: 0.0005770
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005593, upper bound: 0.0005778
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005668, upper bound: 0.0005855
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005719, upper bound: 0.0005891
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005236, upper bound: 0.0005377
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005236, upper bound: 0.0005747
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0005917
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0006041
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005933, upper bound: 0.0005949
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005706, upper bound: 0.0006065
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005269, upper bound: 0.0005896
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.12
Output dim: 5, lower bound: -0.0005269, upper bound: 0.0005912

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030050, 0.0030204
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008472, 0.0008516
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0062511, 0.0062830
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008272, 0.0008315
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0046955, 0.0046717
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013045, 0.0012979
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011841, 0.0011781
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044190, 0.0043966
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034219, 0.0034393
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002967, 0.0002952

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005811, upper bound: 0.0005901
time: 2.02 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005811, upper bound: 0.0005919
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030017, 0.0030201
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008463, 0.0008515
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0062441, 0.0062825
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008263, 0.0008314
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0046951, 0.0046665
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013044, 0.0012965
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011840, 0.0011768
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044186, 0.0043917
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034180, 0.0034390
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002967, 0.0002949

Time for backsubstitution: 0.89 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004178, upper bound: 0.0004155
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004178, upper bound: 0.0004155
time: 0.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030079, 0.0030166
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008480, 0.0008505
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0062569, 0.0062752
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008280, 0.0008304
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0046897, 0.0046761
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013029, 0.0012991
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011827, 0.0011792
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044135, 0.0044007
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034251, 0.0034350
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002964, 0.0002955

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005791, upper bound: 0.0005913
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005791, upper bound: 0.0005886
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030042, 0.0030158
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008470, 0.0008503
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0062493, 0.0062735
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008270, 0.0008302
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0046884, 0.0046703
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013026, 0.0012975
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011823, 0.0011778
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044123, 0.0043953
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034209, 0.0034341
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002963, 0.0002951

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 177

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005677, upper bound: 0.0005789
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005677, upper bound: 0.0005820
time: 1.40 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030728, 0.0030585
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008663, 0.0008623
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0063920, 0.0063623
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008459, 0.0008420
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047548, 0.0047769
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013210, 0.0013272
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011991, 0.0012047
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044748, 0.0044956
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034990, 0.0034827
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003005, 0.0003019

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 102

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005775, upper bound: 0.0005979
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005775, upper bound: 0.0005989
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030709, 0.0030583
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008658, 0.0008622
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0063881, 0.0063619
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008454, 0.0008419
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047545, 0.0047740
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013209, 0.0013264
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011990, 0.0012039
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044745, 0.0044929
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034968, 0.0034825
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003005, 0.0003017

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 207

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004231, upper bound: 0.0004337
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004231, upper bound: 0.0004337
time: 0.98 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0025533, 0.0024973
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007199, 0.0007041
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0053113, 0.0051950
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007029, 0.0006875
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0038824, 0.0039693
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0010786, 0.0011028
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0009791, 0.0010010
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0036538, 0.0037356
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0029074, 0.0028437
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002453, 0.0002508

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 93

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 211

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005300, upper bound: 0.0005798
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005300, upper bound: 0.0005829
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0031471, 0.0031238
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008873, 0.0008807
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0065465, 0.0064982
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008663, 0.0008599
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0048563, 0.0048925
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013492, 0.0013593
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012247, 0.0012338
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0045703, 0.0046044
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035836, 0.0035571
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003069, 0.0003092

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 211

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 25

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005535, upper bound: 0.0005647
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005482, upper bound: 0.0005664
time: 1.47 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0031469, 0.0031239
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008872, 0.0008807
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0065462, 0.0064984
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008663, 0.0008600
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0048565, 0.0048922
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013493, 0.0013592
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012247, 0.0012337
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0045705, 0.0046041
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035834, 0.0035572
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003069, 0.0003092

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 25

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 30

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005785, upper bound: 0.0005835
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005695, upper bound: 0.0005922
time: 1.97 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0029836, 0.0029928
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008412, 0.0008438
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0062066, 0.0062257
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008213, 0.0008239
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0046527, 0.0046384
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0012927, 0.0012887
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011733, 0.0011697
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0043787, 0.0043652
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0033975, 0.0034079
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002940, 0.0002931

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 30
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 111

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005600, upper bound: 0.0005770
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005558, upper bound: 0.0005772
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030650, 0.0030638
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008642, 0.0008638
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0063759, 0.0063732
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008438, 0.0008434
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047630, 0.0047650
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013233, 0.0013238
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012011, 0.0012017
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044825, 0.0044844
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034902, 0.0034887
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003010, 0.0003011

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 205

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005446, upper bound: 0.0005634
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005446, upper bound: 0.0005671
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030678, 0.0030611
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008649, 0.0008630
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0063817, 0.0063676
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008445, 0.0008427
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047588, 0.0047693
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013221, 0.0013251
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012001, 0.0012027
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044785, 0.0044884
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034934, 0.0034857
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003007, 0.0003014

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 185

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005674, upper bound: 0.0005988
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005674, upper bound: 0.0006005
time: 1.32 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030869, 0.0030436
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008703, 0.0008581
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0064214, 0.0063313
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008498, 0.0008378
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047316, 0.0047989
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013146, 0.0013333
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011932, 0.0012102
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044529, 0.0045163
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035151, 0.0034657
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002990, 0.0003033

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005576, upper bound: 0.0005808
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005576, upper bound: 0.0005830
time: 1.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030901, 0.0030406
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008712, 0.0008573
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0064281, 0.0063252
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008507, 0.0008370
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047270, 0.0048039
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013133, 0.0013347
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011921, 0.0012115
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044487, 0.0045210
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035187, 0.0034624
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002987, 0.0003036

Time for backsubstitution: 1.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 205

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005446, upper bound: 0.0005737
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005446, upper bound: 0.0005807
time: 1.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0025423, 0.0024505
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007168, 0.0006909
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0052885, 0.0050975
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0006998, 0.0006746
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0038096, 0.0039523
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0010584, 0.0010981
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0009607, 0.0009967
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0035852, 0.0037195
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0028949, 0.0027904
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002407, 0.0002498

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005111, upper bound: 0.0005756
time: 1.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005111, upper bound: 0.0005740
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0025614, 0.0024289
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0007222, 0.0006848
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0053283, 0.0050527
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0007051, 0.0006686
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0037760, 0.0039821
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0010491, 0.0011063
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0009523, 0.0010042
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0035537, 0.0037476
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0029167, 0.0027658
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002386, 0.0002516

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 185
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 223

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004922, upper bound: 0.0005538
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004922, upper bound: 0.0005550
time: 1.37 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.60 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005811, upper bound: 0.0005901
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005811, upper bound: 0.0005919
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0004178, upper bound: 0.0004155
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0004178, upper bound: 0.0004155
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005791, upper bound: 0.0005913
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005791, upper bound: 0.0005886
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005677, upper bound: 0.0005789
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005677, upper bound: 0.0005820
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005775, upper bound: 0.0005979
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005775, upper bound: 0.0005989
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0004231, upper bound: 0.0004337
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0004231, upper bound: 0.0004337
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005300, upper bound: 0.0005798
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005300, upper bound: 0.0005829
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005535, upper bound: 0.0005647
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005482, upper bound: 0.0005664
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005785, upper bound: 0.0005835
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005695, upper bound: 0.0005922
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005600, upper bound: 0.0005770
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005558, upper bound: 0.0005772
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005446, upper bound: 0.0005634
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005446, upper bound: 0.0005671
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005674, upper bound: 0.0005988
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005674, upper bound: 0.0006005
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005576, upper bound: 0.0005808
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005576, upper bound: 0.0005830
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005446, upper bound: 0.0005737
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005446, upper bound: 0.0005807
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005111, upper bound: 0.0005756
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0005111, upper bound: 0.0005740
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0004922, upper bound: 0.0005538
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.60
Output dim: 5, lower bound: -0.0004922, upper bound: 0.0005550

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0029258, 0.0029639
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008249, 0.0008356
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0060862, 0.0061654
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008054, 0.0008159
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0046077, 0.0045485
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0012801, 0.0012637
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011620, 0.0011471
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0043363, 0.0042806
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0033316, 0.0033750
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002912, 0.0002874

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 113

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 69

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005651, upper bound: 0.0005762
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005651, upper bound: 0.0005740
time: 2.27 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0029500, 0.0029411
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008317, 0.0008292
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0061367, 0.0061181
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008121, 0.0008096
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0045723, 0.0045862
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0012703, 0.0012742
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011531, 0.0011566
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0043030, 0.0043161
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0033592, 0.0033490
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002889, 0.0002898

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005189, upper bound: 0.0005265
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005189, upper bound: 0.0005643
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0029785, 0.0029801
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008398, 0.0008402
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0061960, 0.0061992
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008199, 0.0008204
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0046329, 0.0046305
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0012871, 0.0012865
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011683, 0.0011677
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0043600, 0.0043578
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0033917, 0.0033934
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002928, 0.0002926

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 11

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004044, upper bound: 0.0004031
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004044, upper bound: 0.0004031
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0029713, 0.0029856
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008377, 0.0008418
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0061809, 0.0062106
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008179, 0.0008219
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0046415, 0.0046192
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0012895, 0.0012834
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011705, 0.0011649
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0043681, 0.0043472
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0033835, 0.0033997
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002933, 0.0002919

Time for backsubstitution: 0.99 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 102
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 147

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004720, upper bound: 0.0004704
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004720, upper bound: 0.0004704
time: 1.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0029892, 0.0030009
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008428, 0.0008461
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0062181, 0.0062425
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008229, 0.0008261
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0046653, 0.0046470
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0012962, 0.0012911
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011765, 0.0011719
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0043905, 0.0043734
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034038, 0.0034172
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002948, 0.0002937

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 147

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005478, upper bound: 0.0005635
time: 1.98 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005478, upper bound: 0.0005849
time: 1.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030124, 0.0029749
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008493, 0.0008387
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0062665, 0.0061885
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008293, 0.0008189
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0046249, 0.0046832
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0012849, 0.0013011
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011663, 0.0011810
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0043525, 0.0044074
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034303, 0.0033876
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002923, 0.0002959

Time for backsubstitution: 0.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005651, upper bound: 0.0005865
time: 1.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005651, upper bound: 0.0005874
time: 2.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030998, 0.0030675
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008739, 0.0008649
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0064482, 0.0063811
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008533, 0.0008444
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047688, 0.0048190
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013249, 0.0013389
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012026, 0.0012153
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044880, 0.0045352
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035298, 0.0034930
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003014, 0.0003045

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 170

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004872, upper bound: 0.0005017
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004872, upper bound: 0.0005017
time: 1.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030735, 0.0030679
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008665, 0.0008650
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0063936, 0.0063819
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008461, 0.0008445
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047695, 0.0047781
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013251, 0.0013275
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012028, 0.0012050
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044886, 0.0044968
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034998, 0.0034935
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003014, 0.0003019

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005543, upper bound: 0.0005845
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005543, upper bound: 0.0005874
time: 1.42 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030747, 0.0030703
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008669, 0.0008656
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0063960, 0.0063868
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008464, 0.0008452
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0047731, 0.0047800
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013261, 0.0013280
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0012037, 0.0012054
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044920, 0.0044985
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0035012, 0.0034961
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0003016, 0.0003021

Time for backsubstitution: 0.95 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005386, upper bound: 0.0005668
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 5, lower bound: -0.0005386, upper bound: 0.0005884
time: 1.36 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 4.25 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005651, upper bound: 0.0005762
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005651, upper bound: 0.0005740
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005189, upper bound: 0.0005265
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005189, upper bound: 0.0005643
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0004044, upper bound: 0.0004031
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0004044, upper bound: 0.0004031
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0004720, upper bound: 0.0004704
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0004720, upper bound: 0.0004704
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005478, upper bound: 0.0005635
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005478, upper bound: 0.0005849
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005651, upper bound: 0.0005865
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005651, upper bound: 0.0005874
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0004872, upper bound: 0.0005017
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0004872, upper bound: 0.0005017
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005543, upper bound: 0.0005845
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005543, upper bound: 0.0005874
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005386, upper bound: 0.0005668
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 4.25
Output dim: 5, lower bound: -0.0005386, upper bound: 0.0005884

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0029230, 0.0028842
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008241, 0.0008132
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0060805, 0.0059997
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008047, 0.0007940
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0044838, 0.0045442
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0012457, 0.0012625
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011308, 0.0011460
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0042198, 0.0042766
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0033285, 0.0032843
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002833, 0.0002872

Time for backsubstitution: 0.91 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 46
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 25
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 170
type: DSZ, layer: 1, pos: 178

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 113

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005354, upper bound: 0.0005530
time: 1.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005354, upper bound: 0.0005745
time: 1.92 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0030285, 0.0030128
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008538, 0.0008494
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0062998, 0.0062672
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008337, 0.0008294
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0046837, 0.0047081
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0013013, 0.0013080
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011812, 0.0011873
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0044079, 0.0044308
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0034485, 0.0034307
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002960, 0.0002975

Time for backsubstitution: 0.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 113
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 205

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 93

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004880, upper bound: 0.0005078
time: 1.76 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0004883, upper bound: 0.0005078
time: 1.44 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0094148, -0.0051248, -0.0094148, -0.0051248, -0.0029101, 0.0028865
1: -0.0055930, -0.0043835, -0.0055930, -0.0043835, -0.0008205, 0.0008138
2: -0.0027068, 0.0062173, -0.0027068, 0.0062173, -0.0060537, 0.0060044
3: 0.0012691, 0.0024501, 0.0012691, 0.0024501, -0.0008011, 0.0007946
4: 0.0014454, 0.0081148, 0.0014454, 0.0081148, -0.0044873, 0.0045241
5: 0.9959078, 0.9977608, 0.9959078, 0.9977608, -0.0012467, 0.0012569
6: 0.0041692, 0.0058511, 0.0041692, 0.0058511, -0.0011316, 0.0011409
7: -0.0078228, -0.0015462, -0.0078228, -0.0015462, -0.0042231, 0.0042577
8: -0.0079894, -0.0031043, -0.0079894, -0.0031043, -0.0033138, 0.0032868
9: -0.0037419, -0.0033204, -0.0037419, -0.0033204, -0.0002836, 0.0002859

Time for backsubstitution: 0.92 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 177
type: DSZ, layer: 1, pos: 205
type: DSZ, layer: 1, pos: 178
type: DSZ, layer: 1, pos: 211
type: DSZ, layer: 1, pos: 111
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 11
type: DSZ, layer: 1, pos: 223
type: DSZ, layer: 1, pos: 69
type: DSZ, layer: 1, pos: 147
type: DSZ, layer: 1, pos: 93
type: DSZ, layer: 1, pos: 207
type: DSZ, layer: 1, pos: 119

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005253, upper bound: 0.0005743
time: 1.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 5, lower bound: -0.0005253, upper bound: 0.0005769
time: 1.57 seconds

## Summary of splitting (split count: 9)
- Time for DS candidates: 3.90 seconds
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.90
Output dim: 5, lower bound: -0.0005354, upper bound: 0.0005530
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.90
Output dim: 5, lower bound: -0.0005354, upper bound: 0.0005745
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.90
Output dim: 5, lower bound: -0.0004880, upper bound: 0.0005078
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.90
Output dim: 5, lower bound: -0.0004883, upper bound: 0.0005078
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 10, time: 3.90
Output dim: 5, lower bound: -0.0005253, upper bound: 0.0005743
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 10, time: 3.90
Output dim: 5, lower bound: -0.0005253, upper bound: 0.0005769

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.04 + 345.98 = 349.03 seconds
