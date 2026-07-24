## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist-net_256x4.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0078125
Delta epsilon: 0.00390625
execution index: (1, 2, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.0010952


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0017451, 0.0017451)
1: (-0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004348, 0.0004348)
2: (0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0023044, 0.0023044)
3: (-0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0010489, 0.0010489)
4: (0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004460, 0.0004460)
5: (0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0028984, 0.0028984)
6: (-0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0007356, 0.0007356)
7: (-0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0019033, 0.0019033)
8: (-0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0010009, 0.0010009)
9: (-0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0011606, 0.0011606)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 1.14 + 1.96 = 3.10 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0013690, upper bound: 0.0013690

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013164, upper bound: 0.0013022
time: 1.19 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0013022, upper bound: 0.0013164
time: 1.19 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.51 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.51
Output dim: 0, lower bound: -0.0013164, upper bound: 0.0013022
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.51
Output dim: 0, lower bound: -0.0013022, upper bound: 0.0013164

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0015783, 0.0015646
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0003933, 0.0003899
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0020661, 0.0020841
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0009486, 0.0009404
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0003999, 0.0004034
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0025986, 0.0026213
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0006653, 0.0006595
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0017214, 0.0017064
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009053, 0.0008974
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0010406, 0.0010497

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011561, upper bound: 0.0011484
time: 0.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011561, upper bound: 0.0011484
time: 0.98 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0015646, 0.0015783
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0003899, 0.0003933
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0020841, 0.0020661
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0009404, 0.0009486
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004034, 0.0003999
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0026213, 0.0025986
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0006595, 0.0006653
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0017064, 0.0017214
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0008974, 0.0009053
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0010497, 0.0010406

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011484, upper bound: 0.0011561
time: 0.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011484, upper bound: 0.0011561
time: 0.97 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 3.09 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0011561, upper bound: 0.0011484
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0011561, upper bound: 0.0011484
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0011484, upper bound: 0.0011561
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 3.09
Output dim: 0, lower bound: -0.0011484, upper bound: 0.0011561

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0015779, 0.0015610
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0003932, 0.0003889
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0020612, 0.0020835
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0009483, 0.0009382
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0003989, 0.0004033
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0025925, 0.0026206
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0006651, 0.0006580
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0017209, 0.0017024
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009050, 0.0008953
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0010381, 0.0010494

Time for backsubstitution: 1.03 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008903, upper bound: 0.0008847
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008903, upper bound: 0.0008847
time: 0.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0015746, 0.0015646
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0003924, 0.0003899
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0020661, 0.0020793
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0009464, 0.0009404
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0003999, 0.0004024
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0025986, 0.0026152
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0006638, 0.0006595
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0017174, 0.0017064
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009031, 0.0008974
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0010406, 0.0010472

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008903, upper bound: 0.0008847
time: 0.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008903, upper bound: 0.0008847
time: 0.81 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0015634, 0.0015746
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0003896, 0.0003924
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0020793, 0.0020645
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0009397, 0.0009464
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004024, 0.0003996
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0026152, 0.0025966
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0006590, 0.0006638
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0017051, 0.0017174
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0008967, 0.0009031
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0010472, 0.0010398

Time for backsubstitution: 1.02 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008847, upper bound: 0.0008903
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008847, upper bound: 0.0008903
time: 1.06 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0015610, 0.0015783
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0003889, 0.0003933
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0020841, 0.0020612
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0009382, 0.0009486
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004034, 0.0003989
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0026213, 0.0025925
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0006580, 0.0006653
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0017024, 0.0017214
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0008953, 0.0009053
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0010497, 0.0010381

Time for backsubstitution: 1.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008847, upper bound: 0.0008903
time: 1.05 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008847, upper bound: 0.0008903
time: 1.06 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.21 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.0008903, upper bound: 0.0008847
DS_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.0008903, upper bound: 0.0008847
DS_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.0008903, upper bound: 0.0008847
DS_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.0008903, upper bound: 0.0008847
DS_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.0008847, upper bound: 0.0008903
DS_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.0008847, upper bound: 0.0008903
DS_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.0008847, upper bound: 0.0008903
DS_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 3, time: 3.21
Output dim: 0, lower bound: -0.0008847, upper bound: 0.0008903

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 3.10 + 20.54 = 23.65 seconds
