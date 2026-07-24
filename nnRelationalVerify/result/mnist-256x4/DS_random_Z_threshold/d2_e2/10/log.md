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
execution time: IAR + RelationalAnalysis = 0.79 + 2.52 = 3.31 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -0.1506555, upper bound: 0.1506555

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 72
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 35

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 72

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1415519, upper bound: 0.1415519
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1415519, upper bound: 0.1415519
time: 1.24 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.49 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.49
Output dim: 8, lower bound: -0.1415519, upper bound: 0.1415519
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.49
Output dim: 8, lower bound: -0.1415519, upper bound: 0.1415519

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1415519, upper bound: 0.1415293
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1415293, upper bound: 0.1415519
time: 1.35 seconds

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1414744, upper bound: 0.1412487
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1412487, upper bound: 0.1414744
time: 1.17 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.10 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.10
Output dim: 8, lower bound: -0.1415519, upper bound: 0.1415293
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.10
Output dim: 8, lower bound: -0.1415293, upper bound: 0.1415519
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.10
Output dim: 8, lower bound: -0.1414744, upper bound: 0.1412487
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.10
Output dim: 8, lower bound: -0.1412487, upper bound: 0.1414744

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1276744, upper bound: 0.1276441
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1276744, upper bound: 0.1276441
time: 1.10 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1415214, upper bound: 0.1415495
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1415214, upper bound: 0.1415495
time: 1.24 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1275998, upper bound: 0.1274064
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1275998, upper bound: 0.1274064
time: 0.91 seconds

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 35
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 35

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1412487, upper bound: 0.1414481
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1412138, upper bound: 0.1414744
time: 1.59 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.38 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.38
Output dim: 8, lower bound: -0.1276744, upper bound: 0.1276441
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.38
Output dim: 8, lower bound: -0.1276744, upper bound: 0.1276441
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 8, lower bound: -0.1415214, upper bound: 0.1415495
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 8, lower bound: -0.1415214, upper bound: 0.1415495
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.38
Output dim: 8, lower bound: -0.1275998, upper bound: 0.1274064
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.38
Output dim: 8, lower bound: -0.1275998, upper bound: 0.1274064
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 8, lower bound: -0.1412487, upper bound: 0.1414481
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.38
Output dim: 8, lower bound: -0.1412138, upper bound: 0.1414744

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

Time for backsubstitution: 0.68 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1410229, upper bound: 0.1410593
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1410229, upper bound: 0.1410593
time: 1.25 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1414770, upper bound: 0.1415047
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1414770, upper bound: 0.1415047
time: 1.39 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1412397, upper bound: 0.1414391
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1412397, upper bound: 0.1414391
time: 1.25 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1384205, upper bound: 0.1399358
time: 1.52 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1396795, upper bound: 0.1386836
time: 1.15 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 3.37 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 8, lower bound: -0.1410229, upper bound: 0.1410593
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 8, lower bound: -0.1410229, upper bound: 0.1410593
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 8, lower bound: -0.1414770, upper bound: 0.1415047
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 8, lower bound: -0.1414770, upper bound: 0.1415047
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 8, lower bound: -0.1412397, upper bound: 0.1414391
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 8, lower bound: -0.1412397, upper bound: 0.1414391
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 8, lower bound: -0.1384205, upper bound: 0.1399358
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 3.37
Output dim: 8, lower bound: -0.1396795, upper bound: 0.1386836

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1402686, upper bound: 0.1402821
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1402592, upper bound: 0.1402889
time: 1.40 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1409163, upper bound: 0.1406162
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1406120, upper bound: 0.1409538
time: 1.36 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1403867, upper bound: 0.1405091
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1404809, upper bound: 0.1404107
time: 1.47 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1414330, upper bound: 0.1414896
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1414606, upper bound: 0.1414706
time: 2.67 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1384201, upper bound: 0.1398921
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1397033, upper bound: 0.1386719
time: 1.28 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1405849, upper bound: 0.1407585
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1405849, upper bound: 0.1407695
time: 1.06 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1373466, upper bound: 0.1389572
time: 2.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1374561, upper bound: 0.1388657
time: 1.14 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1396706, upper bound: 0.1386814
time: 2.50 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1396706, upper bound: 0.1386819
time: 1.80 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 5.01 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1402686, upper bound: 0.1402821
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1402592, upper bound: 0.1402889
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1409163, upper bound: 0.1406162
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1406120, upper bound: 0.1409538
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1403867, upper bound: 0.1405091
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1404809, upper bound: 0.1404107
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1414330, upper bound: 0.1414896
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1414606, upper bound: 0.1414706
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1384201, upper bound: 0.1398921
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1397033, upper bound: 0.1386719
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1405849, upper bound: 0.1407585
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1405849, upper bound: 0.1407695
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1373466, upper bound: 0.1389572
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1374561, upper bound: 0.1388657
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1396706, upper bound: 0.1386814
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 5.01
Output dim: 8, lower bound: -0.1396706, upper bound: 0.1386819

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1401651, upper bound: 0.1398720
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1398667, upper bound: 0.1401792
time: 1.36 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391972, upper bound: 0.1393727
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1393382, upper bound: 0.1392560
time: 1.12 seconds

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

Time for backsubstitution: 0.69 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1408374, upper bound: 0.1404105
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1406143, upper bound: 0.1405348
time: 1.33 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1380213, upper bound: 0.1394668
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391104, upper bound: 0.1382239
time: 1.13 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1397453, upper bound: 0.1398580
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1397255, upper bound: 0.1398620
time: 1.11 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1404030, upper bound: 0.1400320
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1401713, upper bound: 0.1403320
time: 1.30 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1275318, upper bound: 0.1276052
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1275318, upper bound: 0.1276052
time: 0.84 seconds

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1413769, upper bound: 0.1411176
time: 1.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1411370, upper bound: 0.1413934
time: 1.15 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379824, upper bound: 0.1394458
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379824, upper bound: 0.1394458
time: 1.00 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1396533, upper bound: 0.1386237
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1396529, upper bound: 0.1386243
time: 1.49 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1404800, upper bound: 0.1403708
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1402681, upper bound: 0.1406555
time: 1.09 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1404800, upper bound: 0.1403728
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1402681, upper bound: 0.1406667
time: 1.32 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367975, upper bound: 0.1383084
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367927, upper bound: 0.1383127
time: 0.97 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1369140, upper bound: 0.1381819
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1369102, upper bound: 0.1382039
time: 1.05 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379006, upper bound: 0.1375269
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1385307, upper bound: 0.1371294
time: 1.18 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1394634, upper bound: 0.1383212
time: 1.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1394023, upper bound: 0.1384504
time: 1.19 seconds

## Summary of splitting (split count: 5)
- Time for DS candidates: 3.44 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1401651, upper bound: 0.1398720
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1398667, upper bound: 0.1401792
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1391972, upper bound: 0.1393727
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1393382, upper bound: 0.1392560
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1408374, upper bound: 0.1404105
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1406143, upper bound: 0.1405348
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1380213, upper bound: 0.1394668
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1391104, upper bound: 0.1382239
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1397453, upper bound: 0.1398580
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1397255, upper bound: 0.1398620
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1404030, upper bound: 0.1400320
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1401713, upper bound: 0.1403320
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1275318, upper bound: 0.1276052
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1275318, upper bound: 0.1276052
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1413769, upper bound: 0.1411176
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1411370, upper bound: 0.1413934
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1379824, upper bound: 0.1394458
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1379824, upper bound: 0.1394458
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1396533, upper bound: 0.1386237
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1396529, upper bound: 0.1386243
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1404800, upper bound: 0.1403708
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1402681, upper bound: 0.1406555
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1404800, upper bound: 0.1403728
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1402681, upper bound: 0.1406667
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1367975, upper bound: 0.1383084
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1367927, upper bound: 0.1383127
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1369140, upper bound: 0.1381819
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1369102, upper bound: 0.1382039
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1379006, upper bound: 0.1375269
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1385307, upper bound: 0.1371294
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1394634, upper bound: 0.1383212
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 6, time: 3.44
Output dim: 8, lower bound: -0.1394023, upper bound: 0.1384504

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1399582, upper bound: 0.1396334
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1398533, upper bound: 0.1396528
time: 1.30 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1382055, upper bound: 0.1390677
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1387692, upper bound: 0.1384597
time: 1.13 seconds

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

Time for backsubstitution: 0.70 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1365221, upper bound: 0.1378348
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376728, upper bound: 0.1366843
time: 1.29 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1392210, upper bound: 0.1391364
time: 0.96 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1392210, upper bound: 0.1391414
time: 1.10 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391580, upper bound: 0.1393288
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1397636, upper bound: 0.1387479
time: 1.19 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1378792, upper bound: 0.1390320
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391205, upper bound: 0.1379237
time: 1.54 seconds

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377792, upper bound: 0.1391575
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376963, upper bound: 0.1392617
time: 1.21 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1374738, upper bound: 0.1371383
time: 1.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1380236, upper bound: 0.1367384
time: 1.65 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391188, upper bound: 0.1392429
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391188, upper bound: 0.1392429
time: 1.07 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1395174, upper bound: 0.1395664
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1394123, upper bound: 0.1396532
time: 1.10 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376719, upper bound: 0.1385095
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1388553, upper bound: 0.1372960
time: 1.20 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1395595, upper bound: 0.1396397
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1395595, upper bound: 0.1396738
time: 1.15 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1396187, upper bound: 0.1399738
time: 1.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1402227, upper bound: 0.1392697
time: 1.83 seconds

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1273234, upper bound: 0.1275140
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1273234, upper bound: 0.1275140
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1378806, upper bound: 0.1390246
time: 1.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377922, upper bound: 0.1393445
time: 2.00 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377542, upper bound: 0.1391425
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376374, upper bound: 0.1392274
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1249994, upper bound: 0.1247586
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1249994, upper bound: 0.1247586
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1249832, upper bound: 0.1247646
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1249832, upper bound: 0.1247646
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1404187, upper bound: 0.1403159
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1404157, upper bound: 0.1403074
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391057, upper bound: 0.1396933
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1393144, upper bound: 0.1395754
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1399094, upper bound: 0.1397864
time: 2.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1399094, upper bound: 0.1397864
time: 2.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1386024, upper bound: 0.1395406
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391490, upper bound: 0.1389780
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1353941, upper bound: 0.1371593
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1356565, upper bound: 0.1365889
time: 1.72 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1228057, upper bound: 0.1233171
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1228057, upper bound: 0.1233171
time: 0.80 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1228065, upper bound: 0.1232341
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1228065, upper bound: 0.1232341
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368682, upper bound: 0.1381553
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368682, upper bound: 0.1381643
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1378016, upper bound: 0.1372031
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376312, upper bound: 0.1374287
time: 1.11 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1384285, upper bound: 0.1371088
time: 1.49 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1385140, upper bound: 0.1370889
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1248368, upper bound: 0.1245596
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1248368, upper bound: 0.1245596
time: 0.82 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376143, upper bound: 0.1372962
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1382599, upper bound: 0.1369003
time: 1.42 seconds

## Summary of splitting (split count: 6)
- Time for DS candidates: 3.45 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1399582, upper bound: 0.1396334
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1398533, upper bound: 0.1396528
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1382055, upper bound: 0.1390677
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1387692, upper bound: 0.1384597
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1365221, upper bound: 0.1378348
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1376728, upper bound: 0.1366843
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1392210, upper bound: 0.1391364
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1392210, upper bound: 0.1391414
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1391580, upper bound: 0.1393288
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1397636, upper bound: 0.1387479
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1378792, upper bound: 0.1390320
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1391205, upper bound: 0.1379237
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1377792, upper bound: 0.1391575
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1376963, upper bound: 0.1392617
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1374738, upper bound: 0.1371383
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1380236, upper bound: 0.1367384
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1391188, upper bound: 0.1392429
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1391188, upper bound: 0.1392429
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1395174, upper bound: 0.1395664
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1394123, upper bound: 0.1396532
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1376719, upper bound: 0.1385095
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1388553, upper bound: 0.1372960
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1395595, upper bound: 0.1396397
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1395595, upper bound: 0.1396738
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1396187, upper bound: 0.1399738
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1402227, upper bound: 0.1392697
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1273234, upper bound: 0.1275140
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1273234, upper bound: 0.1275140
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1378806, upper bound: 0.1390246
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1377922, upper bound: 0.1393445
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1377542, upper bound: 0.1391425
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1376374, upper bound: 0.1392274
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1249994, upper bound: 0.1247586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1249994, upper bound: 0.1247586
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1249832, upper bound: 0.1247646
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1249832, upper bound: 0.1247646
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1404187, upper bound: 0.1403159
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1404157, upper bound: 0.1403074
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1391057, upper bound: 0.1396933
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1393144, upper bound: 0.1395754
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1399094, upper bound: 0.1397864
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1399094, upper bound: 0.1397864
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1386024, upper bound: 0.1395406
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1391490, upper bound: 0.1389780
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1353941, upper bound: 0.1371593
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1356565, upper bound: 0.1365889
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1228057, upper bound: 0.1233171
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1228057, upper bound: 0.1233171
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1228065, upper bound: 0.1232341
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1228065, upper bound: 0.1232341
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1368682, upper bound: 0.1381553
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1368682, upper bound: 0.1381643
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1378016, upper bound: 0.1372031
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1376312, upper bound: 0.1374287
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1384285, upper bound: 0.1371088
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1385140, upper bound: 0.1370889
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1248368, upper bound: 0.1245596
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1248368, upper bound: 0.1245596
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1376143, upper bound: 0.1372962
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 7, time: 3.45
Output dim: 8, lower bound: -0.1382599, upper bound: 0.1369003

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1389092, upper bound: 0.1387231
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1390359, upper bound: 0.1384915
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1388113, upper bound: 0.1387415
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1389377, upper bound: 0.1385270
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1380030, upper bound: 0.1388720
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1380090, upper bound: 0.1388718
time: 2.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1386973, upper bound: 0.1384442
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1387538, upper bound: 0.1383757
time: 1.41 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1364307, upper bound: 0.1375557
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1362176, upper bound: 0.1377592
time: 1.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1375973, upper bound: 0.1363789
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1373483, upper bound: 0.1365938
time: 1.54 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1365520, upper bound: 0.1376222
time: 2.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376888, upper bound: 0.1364551
time: 1.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391438, upper bound: 0.1387925
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1389450, upper bound: 0.1390674
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1366102, upper bound: 0.1378187
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376641, upper bound: 0.1367071
time: 1.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1370419, upper bound: 0.1372441
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1382582, upper bound: 0.1363694
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367681, upper bound: 0.1380240
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368877, upper bound: 0.1378480
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1390189, upper bound: 0.1379026
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391034, upper bound: 0.1378694
time: 1.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377260, upper bound: 0.1391399
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377586, upper bound: 0.1391261
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1365505, upper bound: 0.1382386
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367034, upper bound: 0.1381436
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1372270, upper bound: 0.1367663
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1372052, upper bound: 0.1369075
time: 1.51 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368197, upper bound: 0.1357124
time: 2.45 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1369860, upper bound: 0.1356630
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1389048, upper bound: 0.1389258
time: 1.20 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1388118, upper bound: 0.1390353
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1390439, upper bound: 0.1389836
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1387770, upper bound: 0.1391660
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1394434, upper bound: 0.1393147
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1391888, upper bound: 0.1394909
time: 1.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377700, upper bound: 0.1385567
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1383041, upper bound: 0.1379764
time: 1.27 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1374408, upper bound: 0.1382521
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1372990, upper bound: 0.1382951
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1388131, upper bound: 0.1372750
time: 1.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1388382, upper bound: 0.1372116
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368610, upper bound: 0.1380976
time: 1.17 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1380210, upper bound: 0.1369550
time: 1.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1255304, upper bound: 0.1255888
time: 0.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1255304, upper bound: 0.1255888
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1370374, upper bound: 0.1384481
time: 4.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1380964, upper bound: 0.1371199
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1257325, upper bound: 0.1253524
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1257325, upper bound: 0.1253524
time: 0.78 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1371607, upper bound: 0.1382270
time: 2.29 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1371515, upper bound: 0.1382283
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1363722, upper bound: 0.1382583
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367074, upper bound: 0.1376647
time: 1.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1370247, upper bound: 0.1383123
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1370164, upper bound: 0.1383296
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1369062, upper bound: 0.1384269
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368915, upper bound: 0.1384337
time: 1.39 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376810, upper bound: 0.1387450
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1388633, upper bound: 0.1377359
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1386555, upper bound: 0.1391719
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1392826, upper bound: 0.1386285
time: 1.25 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.72 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1253625, upper bound: 0.1255823
time: 0.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1253625, upper bound: 0.1255823
time: 0.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376624, upper bound: 0.1384683
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1382213, upper bound: 0.1379367
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1388003, upper bound: 0.1388719
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1389990, upper bound: 0.1386805
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1388003, upper bound: 0.1388719
time: 2.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1389990, upper bound: 0.1386805
time: 1.05 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1361838, upper bound: 0.1379566
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1370249, upper bound: 0.1368214
time: 1.24 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1390768, upper bound: 0.1388869
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1390796, upper bound: 0.1388880
time: 1.28 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1211185, upper bound: 0.1216928
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1211185, upper bound: 0.1216928
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1356554, upper bound: 0.1365833
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1356532, upper bound: 0.1365827
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 188
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 188

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368572, upper bound: 0.1381404
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368571, upper bound: 0.1381358
time: 1.22 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 188

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354173, upper bound: 0.1370350
time: 1.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1357568, upper bound: 0.1365076
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1371736, upper bound: 0.1366281
time: 1.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1371736, upper bound: 0.1366319
time: 1.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1230414, upper bound: 0.1230779
time: 0.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1230414, upper bound: 0.1230779
time: 0.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1233668, upper bound: 0.1230904
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1233668, upper bound: 0.1230904
time: 0.83 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1384706, upper bound: 0.1370500
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1384655, upper bound: 0.1370469
time: 1.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1366006, upper bound: 0.1363585
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1366750, upper bound: 0.1362539
time: 2.86 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1378638, upper bound: 0.1365140
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1378638, upper bound: 0.1365140
time: 1.24 seconds

## Summary of splitting (split count: 7)
- Time for DS candidates: 3.26 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1389092, upper bound: 0.1387231
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1390359, upper bound: 0.1384915
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1388113, upper bound: 0.1387415
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1389377, upper bound: 0.1385270
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1380030, upper bound: 0.1388720
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1380090, upper bound: 0.1388718
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1386973, upper bound: 0.1384442
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1387538, upper bound: 0.1383757
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1364307, upper bound: 0.1375557
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1362176, upper bound: 0.1377592
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1375973, upper bound: 0.1363789
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1373483, upper bound: 0.1365938
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1365520, upper bound: 0.1376222
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1376888, upper bound: 0.1364551
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1391438, upper bound: 0.1387925
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1389450, upper bound: 0.1390674
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1366102, upper bound: 0.1378187
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1376641, upper bound: 0.1367071
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1370419, upper bound: 0.1372441
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1382582, upper bound: 0.1363694
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1367681, upper bound: 0.1380240
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1368877, upper bound: 0.1378480
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1390189, upper bound: 0.1379026
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1391034, upper bound: 0.1378694
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1377260, upper bound: 0.1391399
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1377586, upper bound: 0.1391261
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1365505, upper bound: 0.1382386
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1367034, upper bound: 0.1381436
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1372270, upper bound: 0.1367663
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1372052, upper bound: 0.1369075
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1368197, upper bound: 0.1357124
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1369860, upper bound: 0.1356630
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1389048, upper bound: 0.1389258
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1388118, upper bound: 0.1390353
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1390439, upper bound: 0.1389836
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1387770, upper bound: 0.1391660
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1394434, upper bound: 0.1393147
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1391888, upper bound: 0.1394909
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1377700, upper bound: 0.1385567
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1383041, upper bound: 0.1379764
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1374408, upper bound: 0.1382521
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1372990, upper bound: 0.1382951
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1388131, upper bound: 0.1372750
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1388382, upper bound: 0.1372116
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1368610, upper bound: 0.1380976
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1380210, upper bound: 0.1369550
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1255304, upper bound: 0.1255888
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1255304, upper bound: 0.1255888
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1370374, upper bound: 0.1384481
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1380964, upper bound: 0.1371199
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1257325, upper bound: 0.1253524
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1257325, upper bound: 0.1253524
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1371607, upper bound: 0.1382270
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1371515, upper bound: 0.1382283
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1363722, upper bound: 0.1382583
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1367074, upper bound: 0.1376647
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1370247, upper bound: 0.1383123
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1370164, upper bound: 0.1383296
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1369062, upper bound: 0.1384269
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1368915, upper bound: 0.1384337
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1376810, upper bound: 0.1387450
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1388633, upper bound: 0.1377359
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1386555, upper bound: 0.1391719
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1392826, upper bound: 0.1386285
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1253625, upper bound: 0.1255823
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1253625, upper bound: 0.1255823
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1376624, upper bound: 0.1384683
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1382213, upper bound: 0.1379367
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1388003, upper bound: 0.1388719
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1389990, upper bound: 0.1386805
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1388003, upper bound: 0.1388719
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1389990, upper bound: 0.1386805
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1361838, upper bound: 0.1379566
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1370249, upper bound: 0.1368214
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1390768, upper bound: 0.1388869
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1390796, upper bound: 0.1388880
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1211185, upper bound: 0.1216928
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1211185, upper bound: 0.1216928
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1356554, upper bound: 0.1365833
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1356532, upper bound: 0.1365827
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1368572, upper bound: 0.1381404
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1368571, upper bound: 0.1381358
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1354173, upper bound: 0.1370350
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1357568, upper bound: 0.1365076
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1371736, upper bound: 0.1366281
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1371736, upper bound: 0.1366319
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1230414, upper bound: 0.1230779
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1230414, upper bound: 0.1230779
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1233668, upper bound: 0.1230904
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1233668, upper bound: 0.1230904
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1384706, upper bound: 0.1370500
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1384655, upper bound: 0.1370469
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1366006, upper bound: 0.1363585
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1366750, upper bound: 0.1362539
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1378638, upper bound: 0.1365140
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 3.26
Output dim: 8, lower bound: -0.1378638, upper bound: 0.1365140

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1372028, upper bound: 0.1376056
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377859, upper bound: 0.1370328
time: 1.33 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1389590, upper bound: 0.1383019
time: 1.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1387501, upper bound: 0.1384087
time: 1.56 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1386983, upper bound: 0.1386387
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1386981, upper bound: 0.1386324
time: 1.12 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1388859, upper bound: 0.1385103
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1389227, upper bound: 0.1384622
time: 1.63 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377823, upper bound: 0.1385588
time: 1.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377341, upper bound: 0.1386667
time: 1.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377869, upper bound: 0.1385574
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377412, upper bound: 0.1386667
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1384927, upper bound: 0.1382471
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1385036, upper bound: 0.1382470
time: 1.58 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1385548, upper bound: 0.1381772
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1385588, upper bound: 0.1381772
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1363363, upper bound: 0.1372341
time: 1.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1361446, upper bound: 0.1374523
time: 1.15 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1361126, upper bound: 0.1376465
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1361074, upper bound: 0.1376459
time: 1.07 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1373773, upper bound: 0.1360319
time: 1.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1372553, upper bound: 0.1361425
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1356556, upper bound: 0.1354600
time: 1.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1362081, upper bound: 0.1350390
time: 1.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1364601, upper bound: 0.1372509
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1362672, upper bound: 0.1375469
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1374817, upper bound: 0.1360831
time: 1.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1373742, upper bound: 0.1362191
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1373885, upper bound: 0.1376326
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379825, upper bound: 0.1370707
time: 1.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1371692, upper bound: 0.1379065
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377775, upper bound: 0.1373401
time: 1.18 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1363640, upper bound: 0.1375739
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1362282, upper bound: 0.1375913
time: 1.17 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1374907, upper bound: 0.1365368
time: 1.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1374907, upper bound: 0.1365338
time: 1.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368106, upper bound: 0.1369657
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1366545, upper bound: 0.1369806
time: 1.06 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1371153, upper bound: 0.1353502
time: 1.15 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1372113, upper bound: 0.1352688
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367116, upper bound: 0.1379746
time: 1.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367108, upper bound: 0.1379649
time: 1.22 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367689, upper bound: 0.1378302
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368659, upper bound: 0.1378052
time: 1.10 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1388000, upper bound: 0.1375697
time: 1.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1387695, upper bound: 0.1376534
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1382996, upper bound: 0.1371397
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1382996, upper bound: 0.1371546
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.74 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1366061, upper bound: 0.1381224
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367267, upper bound: 0.1380329
time: 1.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1370473, upper bound: 0.1382546
time: 1.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1370399, upper bound: 0.1382777
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1365252, upper bound: 0.1382212
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1365304, upper bound: 0.1381999
time: 1.05 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1366469, upper bound: 0.1381254
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1366834, upper bound: 0.1381170
time: 1.83 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1360764, upper bound: 0.1357395
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1361950, upper bound: 0.1356459
time: 1.23 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1371004, upper bound: 0.1365602
time: 1.39 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1369544, upper bound: 0.1368137
time: 1.37 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1360783, upper bound: 0.1350088
time: 1.33 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1360658, upper bound: 0.1350215
time: 1.30 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 166

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368653, upper bound: 0.1355486
time: 1.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368740, upper bound: 0.1355477
time: 1.13 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1362163, upper bound: 0.1373970
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1373806, upper bound: 0.1361907
time: 1.14 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 52

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1387328, upper bound: 0.1387708
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1385020, upper bound: 0.1389587
time: 1.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1363598, upper bound: 0.1374419
time: 1.27 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1375198, upper bound: 0.1362686
time: 1.85 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1370554, upper bound: 0.1380057
time: 1.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376174, upper bound: 0.1374149
time: 1.16 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1254217, upper bound: 0.1252992
time: 1.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1254217, upper bound: 0.1252992
time: 1.11 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1390849, upper bound: 0.1391457
time: 1.67 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1388480, upper bound: 0.1393860
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 52

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 123

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1352196, upper bound: 0.1369952
time: 1.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1362230, upper bound: 0.1358377
time: 1.36 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 52
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1382685, upper bound: 0.1379612
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1382894, upper bound: 0.1378763
time: 1.09 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1369379, upper bound: 0.1377618
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1369379, upper bound: 0.1377618
time: 1.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1371884, upper bound: 0.1378983
time: 1.13 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1369987, upper bound: 0.1381913
time: 1.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1381721, upper bound: 0.1367135
time: 1.26 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1381617, upper bound: 0.1367203
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1371705, upper bound: 0.1361048
time: 1.29 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377459, upper bound: 0.1358356
time: 1.25 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354092, upper bound: 0.1369744
time: 1.25 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1357596, upper bound: 0.1364342
time: 1.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1374084, upper bound: 0.1363285
time: 1.11 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1374084, upper bound: 0.1363285
time: 1.20 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1364252, upper bound: 0.1377393
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1364186, upper bound: 0.1377421
time: 1.24 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1231821, upper bound: 0.1228434
time: 0.94 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1231821, upper bound: 0.1228434
time: 0.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1361297, upper bound: 0.1373275
time: 1.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1362810, upper bound: 0.1371388
time: 1.90 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.76 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1370343, upper bound: 0.1381043
time: 1.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1370342, upper bound: 0.1381010
time: 1.54 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 166

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1356465, upper bound: 0.1374282
time: 2.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1356425, upper bound: 0.1374349
time: 1.23 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 166
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1366016, upper bound: 0.1376436
time: 1.35 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1366858, upper bound: 0.1375698
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1369227, upper bound: 0.1379816
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368251, upper bound: 0.1382057
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1369145, upper bound: 0.1379867
time: 1.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1368239, upper bound: 0.1382236
time: 1.08 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 250

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367791, upper bound: 0.1382977
time: 1.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1367875, upper bound: 0.1382983
time: 1.15 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 140

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1354567, upper bound: 0.1373291
time: 1.04 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1358075, upper bound: 0.1367195
time: 1.12 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 183

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1235824, upper bound: 0.1238978
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1235824, upper bound: 0.1238978
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 0.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 183

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1377753, upper bound: 0.1368141
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379436, upper bound: 0.1366718
time: 1.26 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 175

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1243854, upper bound: 0.1245489
time: 0.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 8, lower bound: -0.1243854, upper bound: 0.1245489
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 183
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1390686, upper bound: 0.1383672
time: 1.48 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1390140, upper bound: 0.1384039
time: 1.10 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

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

Time for backsubstitution: 0.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 123

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 250

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1375129, upper bound: 0.1384531
time: 1.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376468, upper bound: 0.1384012
time: 1.09 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

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

Time for backsubstitution: 0.80 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 175
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379977, upper bound: 0.1376460
time: 1.31 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1379850, upper bound: 0.1377254
time: 1.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

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

Time for backsubstitution: 0.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6
type: DSZ, layer: 1, pos: 123
type: DSZ, layer: 1, pos: 250
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 140
type: DSZ, layer: 1, pos: 175

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1371174, upper bound: 0.1377588
time: 3.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -0.1376876, upper bound: 0.1371993
time: 2.19 seconds

## Summary of splitting (split count: 8)
- Time for DS candidates: 6.30 seconds
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1372028, upper bound: 0.1376056
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1377859, upper bound: 0.1370328
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1389590, upper bound: 0.1383019
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1387501, upper bound: 0.1384087
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1386983, upper bound: 0.1386387
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1386981, upper bound: 0.1386324
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1388859, upper bound: 0.1385103
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1389227, upper bound: 0.1384622
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1377823, upper bound: 0.1385588
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1377341, upper bound: 0.1386667
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1377869, upper bound: 0.1385574
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1377412, upper bound: 0.1386667
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1384927, upper bound: 0.1382471
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1385036, upper bound: 0.1382470
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1385548, upper bound: 0.1381772
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1385588, upper bound: 0.1381772
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1363363, upper bound: 0.1372341
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1361446, upper bound: 0.1374523
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1361126, upper bound: 0.1376465
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1361074, upper bound: 0.1376459
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1373773, upper bound: 0.1360319
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1372553, upper bound: 0.1361425
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1356556, upper bound: 0.1354600
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1362081, upper bound: 0.1350390
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1364601, upper bound: 0.1372509
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1362672, upper bound: 0.1375469
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1374817, upper bound: 0.1360831
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1373742, upper bound: 0.1362191
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1373885, upper bound: 0.1376326
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1379825, upper bound: 0.1370707
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1371692, upper bound: 0.1379065
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1377775, upper bound: 0.1373401
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1363640, upper bound: 0.1375739
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1362282, upper bound: 0.1375913
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1374907, upper bound: 0.1365368
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1374907, upper bound: 0.1365338
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1368106, upper bound: 0.1369657
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1366545, upper bound: 0.1369806
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1371153, upper bound: 0.1353502
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1372113, upper bound: 0.1352688
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1367116, upper bound: 0.1379746
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1367108, upper bound: 0.1379649
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1367689, upper bound: 0.1378302
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1368659, upper bound: 0.1378052
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1388000, upper bound: 0.1375697
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1387695, upper bound: 0.1376534
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1382996, upper bound: 0.1371397
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1382996, upper bound: 0.1371546
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1366061, upper bound: 0.1381224
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1367267, upper bound: 0.1380329
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1370473, upper bound: 0.1382546
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1370399, upper bound: 0.1382777
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1365252, upper bound: 0.1382212
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1365304, upper bound: 0.1381999
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1366469, upper bound: 0.1381254
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1366834, upper bound: 0.1381170
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1360764, upper bound: 0.1357395
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1361950, upper bound: 0.1356459
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1371004, upper bound: 0.1365602
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1369544, upper bound: 0.1368137
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1360783, upper bound: 0.1350088
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1360658, upper bound: 0.1350215
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1368653, upper bound: 0.1355486
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1368740, upper bound: 0.1355477
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1362163, upper bound: 0.1373970
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1373806, upper bound: 0.1361907
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1387328, upper bound: 0.1387708
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1385020, upper bound: 0.1389587
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1363598, upper bound: 0.1374419
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1375198, upper bound: 0.1362686
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1370554, upper bound: 0.1380057
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1376174, upper bound: 0.1374149
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1254217, upper bound: 0.1252992
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1254217, upper bound: 0.1252992
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1390849, upper bound: 0.1391457
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1388480, upper bound: 0.1393860
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1352196, upper bound: 0.1369952
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1362230, upper bound: 0.1358377
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1382685, upper bound: 0.1379612
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1382894, upper bound: 0.1378763
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1369379, upper bound: 0.1377618
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1369379, upper bound: 0.1377618
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1371884, upper bound: 0.1378983
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1369987, upper bound: 0.1381913
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1381721, upper bound: 0.1367135
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1381617, upper bound: 0.1367203
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1371705, upper bound: 0.1361048
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1377459, upper bound: 0.1358356
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1354092, upper bound: 0.1369744
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1357596, upper bound: 0.1364342
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1374084, upper bound: 0.1363285
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1374084, upper bound: 0.1363285
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1364252, upper bound: 0.1377393
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1364186, upper bound: 0.1377421
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1231821, upper bound: 0.1228434
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1231821, upper bound: 0.1228434
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1361297, upper bound: 0.1373275
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1362810, upper bound: 0.1371388
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1370343, upper bound: 0.1381043
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1370342, upper bound: 0.1381010
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1356465, upper bound: 0.1374282
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1356425, upper bound: 0.1374349
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1366016, upper bound: 0.1376436
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1366858, upper bound: 0.1375698
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1369227, upper bound: 0.1379816
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1368251, upper bound: 0.1382057
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1369145, upper bound: 0.1379867
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1368239, upper bound: 0.1382236
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1367791, upper bound: 0.1382977
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1367875, upper bound: 0.1382983
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1354567, upper bound: 0.1373291
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1358075, upper bound: 0.1367195
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1235824, upper bound: 0.1238978
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1235824, upper bound: 0.1238978
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1377753, upper bound: 0.1368141
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1379436, upper bound: 0.1366718
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1243854, upper bound: 0.1245489
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1243854, upper bound: 0.1245489
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1390686, upper bound: 0.1383672
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1390140, upper bound: 0.1384039
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1375129, upper bound: 0.1384531
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1376468, upper bound: 0.1384012
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1379977, upper bound: 0.1376460
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1379850, upper bound: 0.1377254
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1371174, upper bound: 0.1377588
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 9, time: 6.30
Output dim: 8, lower bound: -0.1376876, upper bound: 0.1371993
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1389990, upper bound: 0.1386805
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1388003, upper bound: 0.1388719
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1389990, upper bound: 0.1386805
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1361838, upper bound: 0.1379566
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1370249, upper bound: 0.1368214
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1390768, upper bound: 0.1388869
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1390796, upper bound: 0.1388880
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1356554, upper bound: 0.1365833
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1356532, upper bound: 0.1365827
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1368572, upper bound: 0.1381404
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1368571, upper bound: 0.1381358
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1354173, upper bound: 0.1370350
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1357568, upper bound: 0.1365076
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1371736, upper bound: 0.1366281
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1371736, upper bound: 0.1366319
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1384706, upper bound: 0.1370500
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1384655, upper bound: 0.1370469
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1366006, upper bound: 0.1363585
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1366750, upper bound: 0.1362539
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1378638, upper bound: 0.1365140
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 8, time: 6.30
Output dim: 8, lower bound: -0.1378638, upper bound: 0.1365140

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 3.31 + 598.62 = 601.93 seconds
