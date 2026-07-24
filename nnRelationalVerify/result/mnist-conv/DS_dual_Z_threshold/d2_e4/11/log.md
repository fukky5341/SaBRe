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
execution time: IAR + RelationalAnalysis = 22.79 + 34.26 = 57.05 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.5992169, upper bound: 0.5992170

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 80
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 80

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5989766, upper bound: 0.5950081
time: 4.03 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5950084, upper bound: 0.5989763
time: 4.14 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 8.48 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 8.48
Output dim: 9, lower bound: -0.5989766, upper bound: 0.5950081
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 8.48
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

Time for backsubstitution: 21.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5947614
time: 4.23 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5987360, upper bound: 0.5944412
time: 3.97 seconds

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

Time for backsubstitution: 21.96 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 66

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 66

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5944413, upper bound: 0.5987356
time: 4.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5947616, upper bound: 0.5984299
time: 4.92 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.69 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.69
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5947614
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.69
Output dim: 9, lower bound: -0.5987360, upper bound: 0.5944412
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.69
Output dim: 9, lower bound: -0.5944413, upper bound: 0.5987356
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.69
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

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.45 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5972198, upper bound: 0.5921991
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5958723, upper bound: 0.5935496
time: 3.77 seconds

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

Time for backsubstitution: 22.06 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.45 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5975252, upper bound: 0.5918790
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5961775, upper bound: 0.5932294
time: 3.89 seconds

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

Time for backsubstitution: 22.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.47 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5932301, upper bound: 0.5961771
time: 4.85 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5918794, upper bound: 0.5975248
time: 4.18 seconds

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

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.43 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5935504, upper bound: 0.5958739
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5921993, upper bound: 0.5972195
time: 4.93 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.03 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.03
Output dim: 9, lower bound: -0.5972198, upper bound: 0.5921991
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.03
Output dim: 9, lower bound: -0.5958723, upper bound: 0.5935496
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.03
Output dim: 9, lower bound: -0.5975252, upper bound: 0.5918790
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.03
Output dim: 9, lower bound: -0.5961775, upper bound: 0.5932294
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.03
Output dim: 9, lower bound: -0.5932301, upper bound: 0.5961771
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.03
Output dim: 9, lower bound: -0.5918794, upper bound: 0.5975248
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.03
Output dim: 9, lower bound: -0.5935504, upper bound: 0.5958739
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.03
Output dim: 9, lower bound: -0.5921993, upper bound: 0.5972195

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.4451184, 1.4397135
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.6679461, 1.6733818
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.2287664, 1.2423620
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1524773, 2.1528797
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6188331, 1.6209977
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4750600, 1.4744551
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7149572, 1.7153454
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.7146378, 1.6988544
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.3385034, 1.3565040
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0279045, 1.0294471

Time for backsubstitution: 21.75 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 758

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5921790, upper bound: 0.5872523
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5921882, upper bound: 0.5872397
time: 3.49 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.4399238, 1.4449084
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.6732714, 1.6680567
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.2423730, 1.2287552
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1527376, 2.1526203
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6208029, 1.6190276
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4745069, 1.4750082
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7151871, 1.7151155
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.6990509, 1.7144413
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.3562317, 1.3387756
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0292730, 1.0280788

Time for backsubstitution: 21.92 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 758

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5909029, upper bound: 0.5885359
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5909138, upper bound: 0.5885266
time: 3.67 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.4449091, 1.4399228
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.6680577, 1.6732702
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.2287550, 1.2423732
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1526184, 2.1527386
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6190281, 1.6208026
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4750085, 1.4745073
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7151151, 1.7151875
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.7144413, 1.6990499
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.3387756, 1.3562312
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0280790, 1.0292728

Time for backsubstitution: 22.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 758

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5924942, upper bound: 0.5869234
time: 3.38 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5925034, upper bound: 0.5869107
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.4397144, 1.4451177
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.6733830, 1.6679451
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.2423620, 1.2287664
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1528788, 2.1524787
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6209984, 1.6188326
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4744554, 1.4750602
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7153449, 1.7149577
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.6988544, 1.7146373
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.3565040, 1.3385031
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0294476, 1.0279043

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 758

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5912184, upper bound: 0.5882063
time: 3.89 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5912292, upper bound: 0.5881978
time: 3.74 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.4451175, 1.4397144
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.6679451, 1.6733828
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.2287664, 1.2423620
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1524782, 2.1528788
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6188326, 1.6209981
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4750600, 1.4744549
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7149577, 1.7153447
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.7146378, 1.6988544
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.3385029, 1.3565042
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0279045, 1.0294476

Time for backsubstitution: 21.95 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 758

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5881969, upper bound: 0.5912306
time: 3.53 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5882053, upper bound: 0.5912180
time: 3.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.4399228, 1.4449091
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.6732705, 1.6680577
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.2423730, 1.2287552
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1527386, 2.1526184
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6208029, 1.6190281
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4745069, 1.4750080
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7151875, 1.7151148
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.6990509, 1.7144413
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.3562312, 1.3387761
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0292730, 1.0280790

Time for backsubstitution: 21.98 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 758

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5869097, upper bound: 0.5925029
time: 6.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5869225, upper bound: 0.5924936
time: 6.95 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.4449086, 1.4399238
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.6680567, 1.6732712
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.2287550, 1.2423732
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1526203, 2.1527371
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6190276, 1.6208031
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4750085, 1.4745071
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7151160, 1.7151866
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.7144413, 1.6990499
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.3387756, 1.3562317
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0280790, 1.0292730

Time for backsubstitution: 22.05 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 758

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5885256, upper bound: 0.5909136
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5885349, upper bound: 0.5909032
time: 4.27 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.4397135, 1.4451184
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.6733820, 1.6679461
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.2423620, 1.2287664
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1528797, 2.1524773
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6209974, 1.6188331
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4744554, 1.4750600
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.7153459, 1.7149568
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.6988544, 1.7146373
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.3565035, 1.3385034
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0294476, 1.0279047

Time for backsubstitution: 22.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 758
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 758

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5872388, upper bound: 0.5921881
time: 3.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5872514, upper bound: 0.5921784
time: 3.93 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.15 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5921790, upper bound: 0.5872523
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5921882, upper bound: 0.5872397
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5909029, upper bound: 0.5885359
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5909138, upper bound: 0.5885266
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5924942, upper bound: 0.5869234
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5925034, upper bound: 0.5869107
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5912184, upper bound: 0.5882063
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5912292, upper bound: 0.5881978
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5881969, upper bound: 0.5912306
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5882053, upper bound: 0.5912180
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5869097, upper bound: 0.5925029
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5869225, upper bound: 0.5924936
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5885256, upper bound: 0.5909136
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5885349, upper bound: 0.5909032
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5872388, upper bound: 0.5921881
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.15
Output dim: 9, lower bound: -0.5872514, upper bound: 0.5921784

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6117530, 1.6109376
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.7741981, 1.7682879
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3292046, 1.3306351
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1276793, 2.1199031
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6008463, 1.6032195
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4669876, 1.4647515
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.6702886, 1.6769428
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8074465, 1.7966218
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4724131, 1.4925644
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0089800, 1.0078163

Time for backsubstitution: 22.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Candidate
type: DSZ, layer: 3, pos: 2228

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5883533, upper bound: 0.5824152
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5875871, upper bound: 0.5827780
time: 3.74 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6111474, 1.6115432
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.7681775, 1.7743089
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3306465, 1.3291929
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1197605, 2.1278214
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6030254, 1.6010408
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4648037, 1.4669356
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.6767840, 1.6704471
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.7968178, 1.8072510
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4922924, 1.4726851
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0076420, 1.0091538

Time for backsubstitution: 22.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Candidate
type: DSZ, layer: 3, pos: 2228

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5881100, upper bound: 0.5824302
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5875864, upper bound: 0.5829651
time: 3.84 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6117530, 1.6109376
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.7741981, 1.7682879
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3292046, 1.3306351
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1276793, 2.1199031
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6008463, 1.6032195
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4669876, 1.4647515
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.6702886, 1.6769428
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.8074465, 1.7966218
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4724131, 1.4925644
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0089800, 1.0078163

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1452
type: DSZ, layer: 3, pos: 2228
type: DSZ, layer: 3, pos: 1109
type: DSZ, layer: 3, pos: 1731
type: DSZ, layer: 3, pos: 765
type: DSZ, layer: 3, pos: 1683
type: DSZ, layer: 3, pos: 766
type: DSZ, layer: 3, pos: 556
type: DSZ, layer: 3, pos: 598
type: DSZ, layer: 3, pos: 2525
type: DSZ, layer: 3, pos: 2460
type: DSZ, layer: 3, pos: 1494
type: DSZ, layer: 3, pos: 563
type: DSZ, layer: 3, pos: 1942
type: DSZ, layer: 3, pos: 311
type: DSZ, layer: 3, pos: 2627
type: DSZ, layer: 3, pos: 1692
type: DSZ, layer: 3, pos: 907
type: DSZ, layer: 3, pos: 921
type: DSZ, layer: 3, pos: 1101
type: DSZ, layer: 3, pos: 2382
type: DSZ, layer: 3, pos: 2607
type: DSZ, layer: 3, pos: 1843
type: DSZ, layer: 3, pos: 913
type: DSZ, layer: 3, pos: 1682
type: DSZ, layer: 3, pos: 1151
type: DSZ, layer: 3, pos: 1920
type: DSZ, layer: 3, pos: 2145
type: DSZ, layer: 3, pos: 1503
type: DSZ, layer: 3, pos: 2544
type: DSZ, layer: 3, pos: 1684
type: DSZ, layer: 3, pos: 2518
type: DSZ, layer: 3, pos: 2825
type: DSZ, layer: 3, pos: 1383
type: DSZ, layer: 3, pos: 46
type: DSZ, layer: 3, pos: 173
type: DSZ, layer: 3, pos: 2909
type: DSZ, layer: 3, pos: 3104
type: DSZ, layer: 3, pos: 1788
type: DSZ, layer: 3, pos: 2152
type: DSZ, layer: 3, pos: 1143
type: DSZ, layer: 3, pos: 1104
type: DSZ, layer: 3, pos: 1492
type: DSZ, layer: 3, pos: 158
type: DSZ, layer: 3, pos: 920
type: DSZ, layer: 3, pos: 1985
type: DSZ, layer: 3, pos: 422
type: DSZ, layer: 3, pos: 1773
type: DSZ, layer: 3, pos: 904
type: DSZ, layer: 3, pos: 897

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 1452

### Candidate
type: DSZ, layer: 3, pos: 2228

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5866516, upper bound: 0.5839290
time: 4.21 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5860962, upper bound: 0.5844544
time: 4.01 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -8.8532009, -6.0920706, -8.8532009, -6.0920706, -1.6111474, 1.6115432
1: -14.2740736, -11.7365341, -14.2740736, -11.7365341, -1.7681775, 1.7743089
2: -8.4603901, -6.5364547, -8.4603901, -6.5364547, -1.3306465, 1.3291929
3: -10.8424759, -8.4118061, -10.8424759, -8.4118061, -2.1197605, 2.1278214
4: -4.4535651, -2.5585072, -4.4535651, -2.5585072, -1.6030254, 1.6010408
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4648037, 1.4669356
6: -10.6835022, -7.8720345, -10.6835022, -7.8720345, -1.6767840, 1.6704471
7: -1.8837423, 0.1428518, -1.8837423, 0.1428518, -1.7968178, 1.8072510
8: -4.4770365, -2.3307447, -4.4770365, -2.3307447, -1.4922924, 1.4726851
9: 1.3801022, 2.6903341, 1.3801022, 2.6903341, -1.0076420, 1.0091538

Time for backsubstitution: 21.99 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.05 + 548.83 = 605.88 seconds
