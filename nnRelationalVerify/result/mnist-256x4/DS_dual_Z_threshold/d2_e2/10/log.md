## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.015625
Delta epsilon: 0.0078125
execution index: (2, 2, 10)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.13558995000000001


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793)
1: (-0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627)
2: (-0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203)
3: (-0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837)
4: (-0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710)
5: (-0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747)
6: (-0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562)
7: (-0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183)
8: (0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069)
9: (-0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.69 + 2.65 = 4.34 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1506555, upper bound: 0.1506555

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1504058, upper bound: 0.1503439
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1503439, upper bound: 0.1504058
time: 1.61 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 3.03 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 3.03
Output dim: 8, lower bound: -0.1504058, upper bound: 0.1503439
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 3.03
Output dim: 8, lower bound: -0.1503439, upper bound: 0.1504058

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1492964, upper bound: 0.1494607
time: 2.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1495181, upper bound: 0.1492375
time: 2.86 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1492375, upper bound: 0.1495181
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1494607, upper bound: 0.1492964
time: 1.55 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 5.60 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 8, lower bound: -0.1492964, upper bound: 0.1494607
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 8, lower bound: -0.1495181, upper bound: 0.1492375
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 8, lower bound: -0.1492375, upper bound: 0.1495181
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 5.60
Output dim: 8, lower bound: -0.1494607, upper bound: 0.1492964

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1492033, upper bound: 0.1493605
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491833, upper bound: 0.1493647
time: 1.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1494228, upper bound: 0.1491333
time: 2.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1494164, upper bound: 0.1491447
time: 1.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491447, upper bound: 0.1494164
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491333, upper bound: 0.1494228
time: 1.76 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1493647, upper bound: 0.1491833
time: 2.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1493605, upper bound: 0.1492033
time: 1.51 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 6.01 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.01
Output dim: 8, lower bound: -0.1492033, upper bound: 0.1493605
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.01
Output dim: 8, lower bound: -0.1491833, upper bound: 0.1493647
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.01
Output dim: 8, lower bound: -0.1494228, upper bound: 0.1491333
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.01
Output dim: 8, lower bound: -0.1494164, upper bound: 0.1491447
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.01
Output dim: 8, lower bound: -0.1491447, upper bound: 0.1494164
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.01
Output dim: 8, lower bound: -0.1491333, upper bound: 0.1494228
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 6.01
Output dim: 8, lower bound: -0.1493647, upper bound: 0.1491833
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 6.01
Output dim: 8, lower bound: -0.1493605, upper bound: 0.1492033

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491703, upper bound: 0.1493431
time: 1.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491848, upper bound: 0.1493387
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491534, upper bound: 0.1493473
time: 2.32 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491646, upper bound: 0.1493430
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1493889, upper bound: 0.1491153
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1494046, upper bound: 0.1491141
time: 2.40 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1493815, upper bound: 0.1491268
time: 1.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1493978, upper bound: 0.1491254
time: 1.68 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491254, upper bound: 0.1493978
time: 3.96 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491268, upper bound: 0.1493815
time: 1.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491141, upper bound: 0.1494046
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491153, upper bound: 0.1493889
time: 2.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1493430, upper bound: 0.1491646
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1493473, upper bound: 0.1491534
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1493387, upper bound: 0.1491848
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1493431, upper bound: 0.1491704
time: 2.26 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 5.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1491703, upper bound: 0.1493431
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1491848, upper bound: 0.1493387
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1491534, upper bound: 0.1493473
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1491646, upper bound: 0.1493430
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1493889, upper bound: 0.1491153
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1494046, upper bound: 0.1491141
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1493815, upper bound: 0.1491268
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1493978, upper bound: 0.1491254
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1491254, upper bound: 0.1493978
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1491268, upper bound: 0.1493815
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1491141, upper bound: 0.1494046
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1491153, upper bound: 0.1493889
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1493430, upper bound: 0.1491646
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1493473, upper bound: 0.1491534
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1493387, upper bound: 0.1491848
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 5.45
Output dim: 8, lower bound: -0.1493431, upper bound: 0.1491704

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1489132, upper bound: 0.1490707
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1489085, upper bound: 0.1490835
time: 2.04 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1489243, upper bound: 0.1490668
time: 1.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1489194, upper bound: 0.1490793
time: 1.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488969, upper bound: 0.1490820
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488875, upper bound: 0.1490905
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1489049, upper bound: 0.1490785
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488986, upper bound: 0.1490861
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491343, upper bound: 0.1488482
time: 2.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491213, upper bound: 0.1488570
time: 1.73 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491454, upper bound: 0.1488460
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491356, upper bound: 0.1488560
time: 1.45 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491221, upper bound: 0.1488663
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491061, upper bound: 0.1488701
time: 1.88 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491354, upper bound: 0.1488639
time: 1.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1491207, upper bound: 0.1488685
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488685, upper bound: 0.1491207
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488639, upper bound: 0.1491354
time: 3.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488701, upper bound: 0.1491061
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488663, upper bound: 0.1491221
time: 1.66 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488560, upper bound: 0.1491356
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488460, upper bound: 0.1491454
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488570, upper bound: 0.1491213
time: 1.66 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488482, upper bound: 0.1491343
time: 1.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490861, upper bound: 0.1488986
time: 1.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490785, upper bound: 0.1489049
time: 5.38 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.64 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490905, upper bound: 0.1488875
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490820, upper bound: 0.1488969
time: 2.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490793, upper bound: 0.1489194
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490668, upper bound: 0.1489243
time: 1.57 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490835, upper bound: 0.1489085
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490707, upper bound: 0.1489132
time: 1.75 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 4.73 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1489132, upper bound: 0.1490707
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1489085, upper bound: 0.1490835
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1489243, upper bound: 0.1490668
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1489194, upper bound: 0.1490793
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1488969, upper bound: 0.1490820
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1488875, upper bound: 0.1490905
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1489049, upper bound: 0.1490785
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1488986, upper bound: 0.1490861
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1491343, upper bound: 0.1488482
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1491213, upper bound: 0.1488570
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1491454, upper bound: 0.1488460
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1491356, upper bound: 0.1488560
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1491221, upper bound: 0.1488663
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1491061, upper bound: 0.1488701
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1491354, upper bound: 0.1488639
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1491207, upper bound: 0.1488685
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1488685, upper bound: 0.1491207
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1488639, upper bound: 0.1491354
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1488701, upper bound: 0.1491061
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1488663, upper bound: 0.1491221
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1488560, upper bound: 0.1491356
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1488460, upper bound: 0.1491454
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1488570, upper bound: 0.1491213
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1488482, upper bound: 0.1491343
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1490861, upper bound: 0.1488986
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1490785, upper bound: 0.1489049
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1490905, upper bound: 0.1488875
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1490820, upper bound: 0.1488969
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1490793, upper bound: 0.1489194
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1490668, upper bound: 0.1489243
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1490835, upper bound: 0.1489085
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 4.73
Output dim: 8, lower bound: -0.1490707, upper bound: 0.1489132

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488283, upper bound: 0.1487442
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1485151, upper bound: 0.1489846
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488217, upper bound: 0.1487472
time: 2.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1485141, upper bound: 0.1489992
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488383, upper bound: 0.1486976
time: 2.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1485684, upper bound: 0.1489808
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488327, upper bound: 0.1486989
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1485684, upper bound: 0.1489951
time: 1.52 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488116, upper bound: 0.1487550
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1484878, upper bound: 0.1489960
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488005, upper bound: 0.1487553
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1484864, upper bound: 0.1490060
time: 1.81 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488188, upper bound: 0.1487100
time: 2.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1485311, upper bound: 0.1489925
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 2.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1488122, upper bound: 0.1487100
time: 2.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1485310, upper bound: 0.1490022
time: 2.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490502, upper bound: 0.1484909
time: 2.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487311, upper bound: 0.1487618
time: 2.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490355, upper bound: 0.1484925
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487303, upper bound: 0.1487715
time: 19.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490608, upper bound: 0.1484596
time: 2.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487895, upper bound: 0.1487596
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490491, upper bound: 0.1484626
time: 1.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487893, upper bound: 0.1487702
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490378, upper bound: 0.1485171
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487193, upper bound: 0.1487798
time: 1.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490196, upper bound: 0.1485172
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487183, upper bound: 0.1487842
time: 1.84 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490500, upper bound: 0.1484852
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487782, upper bound: 0.1487774
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490339, upper bound: 0.1484860
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487769, upper bound: 0.1487831
time: 2.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487831, upper bound: 0.1487769
time: 1.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1484860, upper bound: 0.1490339
time: 2.10 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.59 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487774, upper bound: 0.1487782
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1484852, upper bound: 0.1490500
time: 1.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487842, upper bound: 0.1487183
time: 2.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1485172, upper bound: 0.1490196
time: 1.99 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487798, upper bound: 0.1487193
time: 1.90 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1485171, upper bound: 0.1490378
time: 2.65 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.55 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487702, upper bound: 0.1487893
time: 2.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1484626, upper bound: 0.1490491
time: 1.59 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487596, upper bound: 0.1487895
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1484596, upper bound: 0.1490608
time: 2.84 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487715, upper bound: 0.1487303
time: 2.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1484925, upper bound: 0.1490355
time: 2.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487618, upper bound: 0.1487311
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1484909, upper bound: 0.1490502
time: 1.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.46 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490022, upper bound: 0.1485310
time: 2.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487100, upper bound: 0.1488122
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1489925, upper bound: 0.1485311
time: 2.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487100, upper bound: 0.1488188
time: 2.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1490060, upper bound: 0.1484864
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487553, upper bound: 0.1488005
time: 1.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1489960, upper bound: 0.1484878
time: 1.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487550, upper bound: 0.1488116
time: 2.04 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.56 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1489951, upper bound: 0.1485684
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1486989, upper bound: 0.1488327
time: 2.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1489808, upper bound: 0.1485684
time: 2.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1486976, upper bound: 0.1488383
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1489992, upper bound: 0.1485141
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487472, upper bound: 0.1488217
time: 2.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1489846, upper bound: 0.1485151
time: 1.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1487442, upper bound: 0.1488283
time: 2.49 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 5.88 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1488283, upper bound: 0.1487442
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1485151, upper bound: 0.1489846
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1488217, upper bound: 0.1487472
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1485141, upper bound: 0.1489992
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1488383, upper bound: 0.1486976
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1485684, upper bound: 0.1489808
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1488327, upper bound: 0.1486989
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1485684, upper bound: 0.1489951
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1488116, upper bound: 0.1487550
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1484878, upper bound: 0.1489960
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1488005, upper bound: 0.1487553
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1484864, upper bound: 0.1490060
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1488188, upper bound: 0.1487100
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1485311, upper bound: 0.1489925
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1488122, upper bound: 0.1487100
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1485310, upper bound: 0.1490022
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1490502, upper bound: 0.1484909
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487311, upper bound: 0.1487618
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1490355, upper bound: 0.1484925
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487303, upper bound: 0.1487715
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1490608, upper bound: 0.1484596
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487895, upper bound: 0.1487596
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1490491, upper bound: 0.1484626
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487893, upper bound: 0.1487702
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1490378, upper bound: 0.1485171
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487193, upper bound: 0.1487798
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1490196, upper bound: 0.1485172
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487183, upper bound: 0.1487842
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1490500, upper bound: 0.1484852
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487782, upper bound: 0.1487774
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1490339, upper bound: 0.1484860
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487769, upper bound: 0.1487831
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487831, upper bound: 0.1487769
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1484860, upper bound: 0.1490339
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487774, upper bound: 0.1487782
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1484852, upper bound: 0.1490500
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487842, upper bound: 0.1487183
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1485172, upper bound: 0.1490196
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487798, upper bound: 0.1487193
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1485171, upper bound: 0.1490378
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487702, upper bound: 0.1487893
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1484626, upper bound: 0.1490491
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487596, upper bound: 0.1487895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1484596, upper bound: 0.1490608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487715, upper bound: 0.1487303
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1484925, upper bound: 0.1490355
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487618, upper bound: 0.1487311
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1484909, upper bound: 0.1490502
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1490022, upper bound: 0.1485310
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487100, upper bound: 0.1488122
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1489925, upper bound: 0.1485311
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487100, upper bound: 0.1488188
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1490060, upper bound: 0.1484864
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487553, upper bound: 0.1488005
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1489960, upper bound: 0.1484878
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487550, upper bound: 0.1488116
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1489951, upper bound: 0.1485684
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1486989, upper bound: 0.1488327
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1489808, upper bound: 0.1485684
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1486976, upper bound: 0.1488383
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1489992, upper bound: 0.1485141
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487472, upper bound: 0.1488217
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1489846, upper bound: 0.1485151
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 5.88
Output dim: 8, lower bound: -0.1487442, upper bound: 0.1488283

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471773, upper bound: 0.1478666
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479602, upper bound: 0.1470491
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1468606, upper bound: 0.1481141
time: 1.54 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476393, upper bound: 0.1472707
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471748, upper bound: 0.1478676
time: 2.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479537, upper bound: 0.1470491
time: 2.75 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1468605, upper bound: 0.1481280
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476393, upper bound: 0.1472721
time: 1.40 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.17 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1472279, upper bound: 0.1478225
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479646, upper bound: 0.1469622
time: 1.67 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469734, upper bound: 0.1481106
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476932, upper bound: 0.1472327
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.53 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1472271, upper bound: 0.1478225
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479600, upper bound: 0.1469622
time: 1.78 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469734, upper bound: 0.1481235
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476932, upper bound: 0.1472359
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471671, upper bound: 0.1478770
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479471, upper bound: 0.1470495
time: 2.27 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1468534, upper bound: 0.1481221
time: 1.62 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476134, upper bound: 0.1472758
time: 1.65 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471641, upper bound: 0.1478770
time: 1.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479385, upper bound: 0.1470495
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1468524, upper bound: 0.1481319
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476134, upper bound: 0.1472778
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.58 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1472092, upper bound: 0.1478315
time: 2.34 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479499, upper bound: 0.1469624
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469545, upper bound: 0.1481181
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476577, upper bound: 0.1472361
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1472086, upper bound: 0.1478315
time: 1.88 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479450, upper bound: 0.1469624
time: 1.70 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469545, upper bound: 0.1481274
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476577, upper bound: 0.1472386
time: 1.32 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.13 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1472858, upper bound: 0.1476157
time: 1.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1481752, upper bound: 0.1469038
time: 2.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469845, upper bound: 0.1478945
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1478567, upper bound: 0.1471341
time: 1.59 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.16 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1472826, upper bound: 0.1476169
time: 2.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1481602, upper bound: 0.1469038
time: 1.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469845, upper bound: 0.1478998
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1478567, upper bound: 0.1471350
time: 1.92 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.48 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473430, upper bound: 0.1475877
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1481852, upper bound: 0.1468301
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471055, upper bound: 0.1478934
time: 1.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479150, upper bound: 0.1471129
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473430, upper bound: 0.1475882
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1481728, upper bound: 0.1468304
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471055, upper bound: 0.1478997
time: 1.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479150, upper bound: 0.1471153
time: 2.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1472837, upper bound: 0.1476402
time: 5.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1481639, upper bound: 0.1469175
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469845, upper bound: 0.1479058
time: 2.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1478456, upper bound: 0.1471549
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1472795, upper bound: 0.1476402
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1481461, upper bound: 0.1469175
time: 4.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469845, upper bound: 0.1479088
time: 2.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1478456, upper bound: 0.1471557
time: 2.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473369, upper bound: 0.1476093
time: 4.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1481753, upper bound: 0.1468395
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471001, upper bound: 0.1479051
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479038, upper bound: 0.1471261
time: 1.57 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1473363, upper bound: 0.1476095
time: 1.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1481611, upper bound: 0.1468395
time: 1.55 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471001, upper bound: 0.1479086
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479034, upper bound: 0.1471269
time: 2.06 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471269, upper bound: 0.1479034
time: 1.60 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479086, upper bound: 0.1471001
time: 1.58 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1468395, upper bound: 0.1481611
time: 3.21 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476095, upper bound: 0.1473363
time: 1.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471261, upper bound: 0.1479038
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479051, upper bound: 0.1471001
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1468395, upper bound: 0.1481753
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476093, upper bound: 0.1473369
time: 1.69 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.62 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471558, upper bound: 0.1478456
time: 1.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479088, upper bound: 0.1469845
time: 1.60 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.15 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469175, upper bound: 0.1481461
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476402, upper bound: 0.1472795
time: 1.67 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471549, upper bound: 0.1478456
time: 1.61 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1479058, upper bound: 0.1469845
time: 1.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.14 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1469175, upper bound: 0.1481639
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1476402, upper bound: 0.1472837
time: 1.61 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.18 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1471153, upper bound: 0.1479150
time: 2.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1478997, upper bound: 0.1471055
time: 1.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -0.0649462, 0.0711331, -0.0649462, 0.0711331, -0.1360793, 0.1360793
1: -0.0514351, 0.0405276, -0.0514351, 0.0405276, -0.0919627, 0.0919627
2: -0.0369548, 0.1439655, -0.0369548, 0.1439655, -0.1809203, 0.1809203
3: -0.0710570, 0.0654267, -0.0710570, 0.0654267, -0.1364837, 0.1364837
4: -0.0885529, 0.0217181, -0.0885529, 0.0217181, -0.1102710, 0.1102710
5: -0.0391050, 0.0332697, -0.0391050, 0.0332697, -0.0723747, 0.0723747
6: -0.0304398, 0.0748164, -0.0304398, 0.0748164, -0.1052562, 0.1052562
7: -0.0706228, 0.0447955, -0.0706228, 0.0447955, -0.1154183, 0.1154183
8: 0.7881831, 0.9921900, 0.7881831, 0.9921900, -0.2040069, 0.2040069
9: -0.0352478, 0.1706189, -0.0352478, 0.1706189, -0.2058667, 0.2058667

Time for backsubstitution: 1.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.19 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1468304, upper bound: 0.1481728
time: 1.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1475882, upper bound: 0.1473430
time: 1.54 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 4.88 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471773, upper bound: 0.1478666
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479602, upper bound: 0.1470491
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1468606, upper bound: 0.1481141
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476393, upper bound: 0.1472707
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471748, upper bound: 0.1478676
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479537, upper bound: 0.1470491
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1468605, upper bound: 0.1481280
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476393, upper bound: 0.1472721
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1472279, upper bound: 0.1478225
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479646, upper bound: 0.1469622
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1469734, upper bound: 0.1481106
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476932, upper bound: 0.1472327
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1472271, upper bound: 0.1478225
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479600, upper bound: 0.1469622
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1469734, upper bound: 0.1481235
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476932, upper bound: 0.1472359
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471671, upper bound: 0.1478770
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479471, upper bound: 0.1470495
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1468534, upper bound: 0.1481221
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476134, upper bound: 0.1472758
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471641, upper bound: 0.1478770
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479385, upper bound: 0.1470495
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1468524, upper bound: 0.1481319
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476134, upper bound: 0.1472778
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1472092, upper bound: 0.1478315
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479499, upper bound: 0.1469624
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1469545, upper bound: 0.1481181
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476577, upper bound: 0.1472361
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1472086, upper bound: 0.1478315
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479450, upper bound: 0.1469624
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1469545, upper bound: 0.1481274
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476577, upper bound: 0.1472386
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1472858, upper bound: 0.1476157
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1481752, upper bound: 0.1469038
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1469845, upper bound: 0.1478945
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1478567, upper bound: 0.1471341
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1472826, upper bound: 0.1476169
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1481602, upper bound: 0.1469038
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1469845, upper bound: 0.1478998
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1478567, upper bound: 0.1471350
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1473430, upper bound: 0.1475877
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1481852, upper bound: 0.1468301
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471055, upper bound: 0.1478934
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479150, upper bound: 0.1471129
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1473430, upper bound: 0.1475882
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1481728, upper bound: 0.1468304
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471055, upper bound: 0.1478997
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479150, upper bound: 0.1471153
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1472837, upper bound: 0.1476402
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1481639, upper bound: 0.1469175
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1469845, upper bound: 0.1479058
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1478456, upper bound: 0.1471549
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1472795, upper bound: 0.1476402
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1481461, upper bound: 0.1469175
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1469845, upper bound: 0.1479088
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1478456, upper bound: 0.1471557
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1473369, upper bound: 0.1476093
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1481753, upper bound: 0.1468395
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471001, upper bound: 0.1479051
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479038, upper bound: 0.1471261
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1473363, upper bound: 0.1476095
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1481611, upper bound: 0.1468395
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471001, upper bound: 0.1479086
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479034, upper bound: 0.1471269
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471269, upper bound: 0.1479034
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479086, upper bound: 0.1471001
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1468395, upper bound: 0.1481611
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476095, upper bound: 0.1473363
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471261, upper bound: 0.1479038
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479051, upper bound: 0.1471001
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1468395, upper bound: 0.1481753
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476093, upper bound: 0.1473369
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471558, upper bound: 0.1478456
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479088, upper bound: 0.1469845
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1469175, upper bound: 0.1481461
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476402, upper bound: 0.1472795
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471549, upper bound: 0.1478456
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1479058, upper bound: 0.1469845
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1469175, upper bound: 0.1481639
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1476402, upper bound: 0.1472837
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1471153, upper bound: 0.1479150
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1478997, upper bound: 0.1471055
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1468304, upper bound: 0.1481728
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 4.88
Output dim: 8, lower bound: -0.1475882, upper bound: 0.1473430
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1487596, upper bound: 0.1487895
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1484596, upper bound: 0.1490608
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1487715, upper bound: 0.1487303
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1484925, upper bound: 0.1490355
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1487618, upper bound: 0.1487311
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1484909, upper bound: 0.1490502
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1490022, upper bound: 0.1485310
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1487100, upper bound: 0.1488122
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1489925, upper bound: 0.1485311
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1487100, upper bound: 0.1488188
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1490060, upper bound: 0.1484864
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1487553, upper bound: 0.1488005
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1489960, upper bound: 0.1484878
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1487550, upper bound: 0.1488116
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1489951, upper bound: 0.1485684
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1486989, upper bound: 0.1488327
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1489808, upper bound: 0.1485684
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1486976, upper bound: 0.1488383
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1489992, upper bound: 0.1485141
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1487472, upper bound: 0.1488217
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1489846, upper bound: 0.1485151
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 4.88
Output dim: 8, lower bound: -0.1487442, upper bound: 0.1488283

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 4.34 + 595.97 = 600.31 seconds
