## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 11)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.587232758


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6206393, 1.6206393)
1: (-14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8250618, 1.8250613)
2: (-8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3471718, 1.3471718)
3: (-10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1496143, 2.1496143)
4: (-4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6108632, 1.6108634)
5: (-2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4811802, 1.4811807)
6: (-10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7188563, 1.7188566)
7: (-1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8351016, 1.8351026)
8: (-4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4970908, 1.4970908)
9: (1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0213950, 1.0213950)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.23 + 34.60 = 58.82 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.5992169, upper bound: 0.5992170

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5989766, upper bound: 0.5950081
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5950084, upper bound: 0.5989763
time: 3.90 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.63 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.63
Output dim: 9, lower bound: -0.5989766, upper bound: 0.5950081
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.63
Output dim: 9, lower bound: -0.5950084, upper bound: 0.5989763

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6206450, 1.6206446
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8250608, 1.8250599
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3471723, 1.3471718
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1496234, 2.1496248
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6108623, 1.6108620
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4811816, 1.4811821
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7188611, 1.7188618
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8351026, 1.8351026
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4970932, 1.4970930
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0213945, 1.0213944

Time for backsubstitution: 23.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5947614
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5987360, upper bound: 0.5944412
time: 3.75 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6206441, 1.6206450
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8250599, 1.8250608
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3471718, 1.3471720
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1496243, 2.1496234
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6108623, 1.6108625
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4811816, 1.4811819
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7188621, 1.7188611
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8351026, 1.8351026
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4970932, 1.4970932
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0213940, 1.0213946

Time for backsubstitution: 23.12 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5944413, upper bound: 0.5987356
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5947616, upper bound: 0.5984299
time: 4.81 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 32.30 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 32.30
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5947614
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 32.30
Output dim: 9, lower bound: -0.5987360, upper bound: 0.5944412
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 32.30
Output dim: 9, lower bound: -0.5944413, upper bound: 0.5987356
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 32.30
Output dim: 9, lower bound: -0.5947616, upper bound: 0.5984299

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6223626, 1.6221528
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8241572, 1.8242676
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3472500, 1.3472385
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1506224, 2.1507649
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6093001, 1.6094942
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4807663, 1.4807146
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7176104, 1.7177694
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8347578, 1.8345623
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4947343, 1.4950066
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0198092, 1.0199835

Time for backsubstitution: 23.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2544

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5909624, upper bound: 0.5932805
time: 3.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5969557, upper bound: 0.5872899
time: 3.60 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6221528, 1.6223617
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8242688, 1.8241560
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3472385, 1.3472497
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1507635, 2.1506238
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6094947, 1.6092992
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4807138, 1.4807668
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7177687, 1.7176116
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8345623, 1.8347578
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4950070, 1.4947340
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0199838, 1.0198090

Time for backsubstitution: 23.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1773

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5984495, upper bound: 0.5931958
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5975124, upper bound: 0.5941347
time: 3.88 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6223617, 1.6221538
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8241563, 1.8242686
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3472500, 1.3472385
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1506233, 2.1507640
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6092992, 1.6094947
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4807663, 1.4807143
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7176113, 1.7177687
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8347578, 1.8345623
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4947343, 1.4950068
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0198092, 1.0199838

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1682

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1143

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5906359, upper bound: 0.5949583
time: 5.16 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5906359, upper bound: 0.5949583
time: 4.74 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6221528, 1.6223626
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8242679, 1.8241570
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3472385, 1.3472497
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1507645, 2.1506224
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6094942, 1.6092997
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4807138, 1.4807665
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7177696, 1.7176106
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8345623, 1.8347578
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4950066, 1.4947343
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0199838, 1.0198095

Time for backsubstitution: 23.42 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1942

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1773

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5944641, upper bound: 0.5971974
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5935261, upper bound: 0.5981340
time: 6.87 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 34.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.31
Output dim: 9, lower bound: -0.5909624, upper bound: 0.5932805
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.31
Output dim: 9, lower bound: -0.5969557, upper bound: 0.5872899
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.31
Output dim: 9, lower bound: -0.5984495, upper bound: 0.5931958
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.31
Output dim: 9, lower bound: -0.5975124, upper bound: 0.5941347
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.31
Output dim: 9, lower bound: -0.5906359, upper bound: 0.5949583
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.31
Output dim: 9, lower bound: -0.5906359, upper bound: 0.5949583
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 34.31
Output dim: 9, lower bound: -0.5944641, upper bound: 0.5971974
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 34.31
Output dim: 9, lower bound: -0.5935261, upper bound: 0.5981340

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6221347, 1.6220016
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8222556, 1.8215282
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3472567, 1.3472273
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1497707, 2.1496148
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6089454, 1.6090941
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4802079, 1.4785521
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7165275, 1.7157555
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8326368, 1.8333006
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4941034, 1.4938641
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0163598, 1.0188427

Time for backsubstitution: 22.79 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2518

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 920

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5784531, upper bound: 0.5761823
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5784531, upper bound: 0.5761823
time: 3.59 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6222119, 1.6219244
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8214178, 1.8223660
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3472385, 1.3472459
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1494732, 2.1499133
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6088996, 1.6091392
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4786038, 1.4801564
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7155967, 1.7166858
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8334951, 1.8324413
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4935918, 1.4943755
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0186687, 1.0165343

Time for backsubstitution: 22.81 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 1942

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 920

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5798695, upper bound: 0.5747773
time: 4.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5798695, upper bound: 0.5747773
time: 4.35 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6209521, 1.6213102
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8231120, 1.8226860
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3471441, 1.3467705
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1502647, 2.1499724
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6089859, 1.6086452
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4805288, 1.4808981
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7164617, 1.7161868
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8320913, 1.8335156
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4949656, 1.4947143
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0199673, 1.0197673

Time for backsubstitution: 22.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1843

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1492

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5947398, upper bound: 0.5931947
time: 3.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5984432, upper bound: 0.5897734
time: 3.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6211019, 1.6211605
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8227983, 1.8229997
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3467588, 1.3471553
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1501131, 2.1501250
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6088409, 1.6087902
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4808455, 1.4805818
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7163439, 1.7163048
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8333197, 1.8322868
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4949870, 1.4946930
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0199425, 1.0197923

Time for backsubstitution: 22.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 921

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1101

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5962966, upper bound: 0.5929151
time: 4.05 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5962913, upper bound: 0.5929232
time: 4.00 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6218414, 1.6216950
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8232203, 1.8240805
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3456221, 1.3462982
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1486635, 2.1504741
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6088295, 1.6089520
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4792833, 1.4781108
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7175064, 1.7176759
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8324003, 1.8320122
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4945841, 1.4948456
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0183277, 1.0186796

Time for backsubstitution: 22.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 311

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5904655, upper bound: 0.5946652
time: 6.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5903429, upper bound: 0.5947832
time: 3.86 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6219034, 1.6216331
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8239679, 1.8233330
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3463097, 1.3456113
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1503344, 2.1488037
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6087561, 1.6090248
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4781628, 1.4792314
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7175188, 1.7176638
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8322086, 1.8322048
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4945726, 1.4948568
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0185046, 1.0185025

Time for backsubstitution: 22.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 3104

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 46

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5900527, upper bound: 0.5944804
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5901577, upper bound: 0.5943754
time: 4.69 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6209512, 1.6213112
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8231111, 1.8226869
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3471441, 1.3467705
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1502662, 2.1499715
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6089854, 1.6086457
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4805298, 1.4808979
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7164621, 1.7161858
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8320913, 1.8335156
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4949651, 1.4947145
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0199668, 1.0197678

Time for backsubstitution: 25.32 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 904

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1942

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5773473, upper bound: 0.5883859
time: 4.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5857286, upper bound: 0.5800168
time: 3.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6211009, 1.6211615
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8227973, 1.8230007
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3467588, 1.3471553
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1501141, 2.1501241
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6088405, 1.6087906
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4808464, 1.4805815
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7163444, 1.7163038
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8333197, 1.8322868
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4949865, 1.4946933
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0199420, 1.0197928

Time for backsubstitution: 25.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 766

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 921

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5907729, upper bound: 0.5971784
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5926146, upper bound: 0.5953738
time: 3.72 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 33.21 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5784531, upper bound: 0.5761823
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5784531, upper bound: 0.5761823
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5798695, upper bound: 0.5747773
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5798695, upper bound: 0.5747773
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5947398, upper bound: 0.5931947
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5984432, upper bound: 0.5897734
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5962966, upper bound: 0.5929151
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5962913, upper bound: 0.5929232
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5904655, upper bound: 0.5946652
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5903429, upper bound: 0.5947832
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5900527, upper bound: 0.5944804
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5901577, upper bound: 0.5943754
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5773473, upper bound: 0.5883859
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5857286, upper bound: 0.5800168
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5907729, upper bound: 0.5971784
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 33.21
Output dim: 9, lower bound: -0.5926146, upper bound: 0.5953738

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6187134, 1.6182079
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8175855, 1.8158581
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3482895, 1.3478878
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1406446, 2.1378670
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6038980, 1.6056693
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4916162, 1.4905457
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7212496, 1.7203839
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8328505, 1.8329167
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4949999, 1.4949114
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0198460, 1.0198069

Time for backsubstitution: 25.30 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1151

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5933162, upper bound: 0.5919285
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5934988, upper bound: 0.5918108
time: 3.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6179996, 1.6189213
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8159709, 1.8174725
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3478765, 1.3483007
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1380067, 2.1405044
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6058650, 1.6037025
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4904928, 1.4916685
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7205410, 1.7210925
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8327208, 1.8330455
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4951839, 1.4947269
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0199819, 1.0196710

Time for backsubstitution: 25.29 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 897
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1109

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 598

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5953703, upper bound: 0.5869465
time: 4.10 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5953799, upper bound: 0.5869464
time: 4.01 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6207156, 1.6208096
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.8298264, 1.8280628
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3396888, 1.3384202
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1360631, 2.1358032
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6063523, 1.6052263
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4243412, 1.4289584
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7245622, 1.7225382
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8017745, 1.8085871
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4540424, 1.4446387
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0155420, 1.0154603

Time for backsubstitution: 25.36 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.82 + 544.16 = 602.98 seconds
