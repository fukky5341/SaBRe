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
execution time: IAR + RelationalAnalysis = 0.94 + 1.89 = 2.83 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.0013690, upper bound: 0.0013690

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 182
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 182

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012731, upper bound: 0.0012711
time: 1.06 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012711, upper bound: 0.0012731
time: 0.98 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 2.04 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 2.04
Output dim: 0, lower bound: -0.0012731, upper bound: 0.0012711
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 2.04
Output dim: 0, lower bound: -0.0012711, upper bound: 0.0012731

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0017360, 0.0017189
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004326, 0.0004283
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0022698, 0.0022924
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0010434, 0.0010331
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004393, 0.0004437
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0028548, 0.0028832
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0007318, 0.0007246
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0018933, 0.0018747
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009957, 0.0009859
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0011432, 0.0011546

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Relational analysis ABCD of DS_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 119

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012681, upper bound: 0.0012662
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012681, upper bound: 0.0012661
time: 1.04 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0017189, 0.0017451
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004283, 0.0004348
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0023044, 0.0022698
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0010331, 0.0010489
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004460, 0.0004393
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0028984, 0.0028548
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0007246, 0.0007356
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0018747, 0.0019033
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009859, 0.0010009
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0011606, 0.0011432

Time for backsubstitution: 0.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 119
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009911, upper bound: 0.0009911
time: 1.03 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009911, upper bound: 0.0009911
time: 1.02 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 4.71 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 4.71
Output dim: 0, lower bound: -0.0012681, upper bound: 0.0012662
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 4.71
Output dim: 0, lower bound: -0.0012681, upper bound: 0.0012661
DS_DSZ2_DSZ1, status: Status.VERIFIED, split count: 2, time: 4.71
Output dim: 0, lower bound: -0.0009911, upper bound: 0.0009911
DS_DSZ2_DSZ2, status: Status.VERIFIED, split count: 2, time: 4.71
Output dim: 0, lower bound: -0.0009911, upper bound: 0.0009911

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0017369, 0.0017147
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004328, 0.0004273
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0022642, 0.0022935
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0010439, 0.0010306
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004382, 0.0004439
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0028478, 0.0028847
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0007322, 0.0007228
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0018943, 0.0018701
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009962, 0.0009835
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0011404, 0.0011552

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 229

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 210

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 249

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012283, upper bound: 0.0012079
time: 1.08 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0012095, upper bound: 0.0012259
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0017318, 0.0017176
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004315, 0.0004280
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0022681, 0.0022868
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0010408, 0.0010324
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004390, 0.0004426
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0028527, 0.0028761
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0007300, 0.0007240
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0018887, 0.0018733
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009933, 0.0009852
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0011423, 0.0011517

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011690, upper bound: 0.0011689
time: 1.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011690, upper bound: 0.0011689
time: 1.28 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 3.40 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -0.0012283, upper bound: 0.0012079
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -0.0012095, upper bound: 0.0012259
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -0.0011690, upper bound: 0.0011689
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 3.40
Output dim: 0, lower bound: -0.0011690, upper bound: 0.0011689

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0016810, 0.0016390
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004189, 0.0004084
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0021643, 0.0022197
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0010103, 0.0009851
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004189, 0.0004296
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0027222, 0.0027918
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0007086, 0.0006909
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0018334, 0.0017876
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009641, 0.0009401
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0010901, 0.0011180

Time for backsubstitution: 0.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011280, upper bound: 0.0011096
time: 1.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011280, upper bound: 0.0011096
time: 1.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0016613, 0.0016563
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004139, 0.0004127
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0021871, 0.0021937
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0009985, 0.0009955
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004233, 0.0004246
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0027508, 0.0027591
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0007003, 0.0006982
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0018118, 0.0018064
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009528, 0.0009500
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0011015, 0.0011048

Time for backsubstitution: 0.90 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 229
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 242

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 229

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011092, upper bound: 0.0011280
time: 1.00 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.0011092, upper bound: 0.0011280
time: 1.00 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0016960, 0.0016869
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004226, 0.0004203
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0022275, 0.0022396
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0010194, 0.0010139
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004311, 0.0004335
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0028016, 0.0028168
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0007149, 0.0007111
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0018498, 0.0018398
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009728, 0.0009675
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0011219, 0.0011280

Time for backsubstitution: 0.87 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 249
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 70

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 132

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008174, upper bound: 0.0008174
time: 0.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008174, upper bound: 0.0008174
time: 0.78 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0017010, 0.0017176
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004238, 0.0004280
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0022681, 0.0022461
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0010223, 0.0010324
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004390, 0.0004347
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0028527, 0.0028251
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0007170, 0.0007240
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0018552, 0.0018733
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009756, 0.0009852
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0011423, 0.0011313

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 249

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008174, upper bound: 0.0008174
time: 0.79 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0008174, upper bound: 0.0008174
time: 0.79 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 2.40 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0011280, upper bound: 0.0011096
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0011280, upper bound: 0.0011096
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0011092, upper bound: 0.0011280
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0011092, upper bound: 0.0011280
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0008174, upper bound: 0.0008174
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0008174, upper bound: 0.0008174
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0008174, upper bound: 0.0008174
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 2.40
Output dim: 0, lower bound: -0.0008174, upper bound: 0.0008174

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0016503, 0.0016084
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004112, 0.0004008
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0021239, 0.0021791
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0009919, 0.0009667
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004111, 0.0004218
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0026713, 0.0027408
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0006956, 0.0006780
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0017998, 0.0017542
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009465, 0.0009225
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0010697, 0.0010975

Time for backsubstitution: 0.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 70

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010810, upper bound: 0.0010536
time: 1.16 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010725, upper bound: 0.0010633
time: 1.08 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0016504, 0.0016390
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004112, 0.0004084
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0021643, 0.0021793
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0009919, 0.0009851
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004189, 0.0004218
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0027222, 0.0027410
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0006957, 0.0006909
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0018000, 0.0017876
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009466, 0.0009401
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0010901, 0.0010976

Time for backsubstitution: 0.83 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 89

### Candidate
type: DSZ, layer: 1, pos: 210

### Candidate
type: DSZ, layer: 1, pos: 242

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0010264, upper bound: 0.0009803
time: 1.09 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0009983, upper bound: 0.0010087
time: 1.29 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0016284, 0.0016257
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004058, 0.0004051
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0021467, 0.0021503
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0009787, 0.0009771
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004155, 0.0004162
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0026999, 0.0027045
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0006864, 0.0006853
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0017760, 0.0017730
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009340, 0.0009324
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0010812, 0.0010830

Time for backsubstitution: 0.84 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 210
type: DSZ, layer: 1, pos: 89

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 167

### Candidate
type: DSZ, layer: 1, pos: 132

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007680, upper bound: 0.0007696
time: 0.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007680, upper bound: 0.0007696
time: 0.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 0.9906954, 0.9931488, 0.9906954, 0.9931488, -0.0016306, 0.0016563
1: -0.0035824, -0.0029711, -0.0035824, -0.0029711, -0.0004063, 0.0004127
2: 0.0056912, 0.0089309, 0.0056912, 0.0089309, -0.0021871, 0.0021532
3: -0.0053381, -0.0038635, -0.0053381, -0.0038635, -0.0009801, 0.0009955
4: 0.0016294, 0.0022564, 0.0016294, 0.0022564, -0.0004233, 0.0004168
5: 0.0061174, 0.0101921, 0.0061174, 0.0101921, -0.0027508, 0.0027082
6: -0.0010460, -0.0000118, -0.0010460, -0.0000118, -0.0006874, 0.0006982
7: -0.0058441, -0.0031683, -0.0058441, -0.0031683, -0.0017784, 0.0018064
8: -0.0026375, -0.0012303, -0.0026375, -0.0012303, -0.0009353, 0.0009500
9: -0.0004373, 0.0011944, -0.0004373, 0.0011944, -0.0011015, 0.0010845

Time for backsubstitution: 0.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 109
type: DSZ, layer: 1, pos: 242
type: DSZ, layer: 1, pos: 132
type: DSZ, layer: 1, pos: 167
type: DSZ, layer: 1, pos: 70
type: DSZ, layer: 1, pos: 89
type: DSZ, layer: 1, pos: 210

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 109

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007680, upper bound: 0.0007696
time: 0.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.0007680, upper bound: 0.0007696
time: 0.89 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 2.78 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.0010810, upper bound: 0.0010536
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.0010725, upper bound: 0.0010633
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.0010264, upper bound: 0.0009803
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.0009983, upper bound: 0.0010087
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.0007680, upper bound: 0.0007696
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.0007680, upper bound: 0.0007696
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.0007680, upper bound: 0.0007696
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 2.78
Output dim: 0, lower bound: -0.0007680, upper bound: 0.0007696

## DS Result
status: Status.VERIFIED
execution time: (base) + (ds) = 2.83 + 47.43 = 50.26 seconds
