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
execution time: IAR + RelationalAnalysis = 22.76 + 33.87 = 56.63 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.5992169, upper bound: 0.5992170

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 66
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5991908
time: 3.69 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5989774, upper bound: 0.5989777
time: 3.69 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.65 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.65
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5991908
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.65
Output dim: 9, lower bound: -0.5989774, upper bound: 0.5989777

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -8.8529787, -6.0920897, -8.8531837, -6.0920792, -1.6221910, 1.6205988
1: -14.2732525, -11.7366552, -14.2737465, -11.7365398, -1.8242221, 1.8246081
2: -8.4603930, -6.5364151, -8.4603891, -6.5365129, -1.3471165, 1.3472137
3: -10.8424587, -8.4108410, -10.8424673, -8.4118624, -2.1495357, 2.1505742
4: -4.4520521, -2.5586677, -4.4529800, -2.5585072, -1.6093206, 1.6100144
5: -2.9136362, -1.0502625, -2.9139047, -1.0502214, -1.4807720, 1.4808891
6: -10.6823788, -7.8721609, -10.6830616, -7.8720360, -1.7177410, 1.7182813
7: -1.8836951, 0.1442804, -1.8837228, 0.1428499, -1.8349252, 1.8347740
8: -4.4790006, -2.3308189, -4.4770288, -2.3307796, -1.4962063, 1.4950476
9: 1.3788481, 2.6902061, 1.3801022, 2.6902974, -1.0207317, 1.0201759

Time for backsubstitution: 21.76 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 66
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5986718
time: 4.04 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5989777
time: 3.75 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -8.8532000, -6.0922627, -8.8532009, -6.0920706, -1.6206393, 1.6223526
1: -14.2740726, -11.7365370, -14.2740736, -11.7365341, -1.8242683, 1.8250608
2: -8.4603796, -6.5364571, -8.4603901, -6.5364547, -1.3472371, 1.3471692
3: -10.8423510, -8.4118052, -10.8424759, -8.4118061, -2.1507540, 2.1496139
4: -4.4535594, -2.5585070, -4.4535651, -2.5585072, -1.6094913, 1.6108634
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4807119, 1.4811795
6: -10.6835003, -7.8720369, -10.6835022, -7.8720345, -1.7177644, 1.7188559
7: -1.8835671, 0.1428504, -1.8837423, 0.1428518, -1.8345623, 1.8351021
8: -4.4770365, -2.3309865, -4.4770365, -2.3307447, -1.4970908, 1.4947314
9: 1.3801057, 2.6901963, 1.3801022, 2.6903341, -1.0213907, 1.0198097

Time for backsubstitution: 22.07 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5989774, upper bound: 0.5986723
time: 4.29 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5989774, upper bound: 0.5989776
time: 3.70 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.34 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 30.34
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5986718
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.34
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5989777
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.34
Output dim: 9, lower bound: -0.5989774, upper bound: 0.5986723
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.34
Output dim: 9, lower bound: -0.5989774, upper bound: 0.5989776

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -8.8529787, -6.0920897, -8.8529787, -6.0920897, -1.6221800, 1.6221800
1: -14.2732525, -11.7366552, -14.2732525, -11.7366552, -1.8241067, 1.8241072
2: -8.4603930, -6.5364151, -8.4603930, -6.5364151, -1.3472147, 1.3472147
3: -10.8424587, -8.4108410, -10.8424587, -8.4108410, -2.1505666, 2.1505661
4: -4.4520521, -2.5586677, -4.4520521, -2.5586677, -1.6091595, 1.6091592
5: -2.9136362, -1.0502625, -2.9136362, -1.0502625, -1.4807301, 1.4807303
6: -10.6823788, -7.8721609, -10.6823788, -7.8721609, -1.7176204, 1.7176204
7: -1.8836951, 0.1442804, -1.8836951, 0.1442804, -1.8347445, 1.8347449
8: -4.4790006, -2.3308189, -4.4790006, -2.3308189, -1.4949775, 1.4949775
9: 1.3788481, 2.6902061, 1.3788481, 2.6902061, -1.0200329, 1.0200329

Time for backsubstitution: 22.01 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5985977, upper bound: 0.5948069
time: 3.77 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5988801
time: 4.02 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -8.8529787, -6.0920897, -8.8532000, -6.0922627, -1.6220093, 1.6206188
1: -14.2732525, -11.7366552, -14.2740726, -11.7365370, -1.8242273, 1.8249402
2: -8.4603930, -6.5364151, -8.4603796, -6.5364571, -1.3471694, 1.3472030
3: -10.8424587, -8.4108410, -10.8423510, -8.4118052, -2.1495991, 2.1504555
4: -4.4520521, -2.5586677, -4.4535594, -2.5585070, -1.6093221, 1.6106961
5: -2.9136362, -1.0502625, -2.9140515, -1.0502224, -1.4807711, 1.4811385
6: -10.6823788, -7.8721609, -10.6835003, -7.8720369, -1.7177410, 1.7187333
7: -1.8836951, 0.1442804, -1.8835671, 0.1428504, -1.8349266, 1.8349204
8: -4.4790006, -2.3308189, -4.4770365, -2.3309865, -1.4970155, 1.4950523
9: 1.3788481, 2.6902061, 1.3801057, 2.6901963, -1.0212481, 1.0201719

Time for backsubstitution: 21.84 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5985977, upper bound: 0.5948781
time: 3.77 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5989485
time: 4.39 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -8.8532000, -6.0922627, -8.8529787, -6.0920897, -1.6206183, 1.6220098
1: -14.2740726, -11.7365370, -14.2732525, -11.7366552, -1.8249397, 1.8242271
2: -8.4603796, -6.5364571, -8.4603930, -6.5364151, -1.3472033, 1.3471696
3: -10.8423510, -8.4118052, -10.8424587, -8.4108410, -2.1504560, 2.1495996
4: -4.4535594, -2.5585070, -4.4520521, -2.5586677, -1.6106958, 1.6093218
5: -2.9140515, -1.0502224, -2.9136362, -1.0502625, -1.4811382, 1.4807711
6: -10.6835003, -7.8720369, -10.6823788, -7.8721609, -1.7187333, 1.7177410
7: -1.8835671, 0.1428504, -1.8836951, 0.1442804, -1.8349199, 1.8349266
8: -4.4770365, -2.3309865, -4.4790006, -2.3308189, -1.4950523, 1.4970155
9: 1.3801057, 2.6901963, 1.3788481, 2.6902061, -1.0201719, 1.0212482

Time for backsubstitution: 21.99 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80
type: A, layer: 1, pos: 80

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5946488, upper bound: 0.5985988
time: 3.49 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5987360, upper bound: 0.5984304
time: 3.98 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -8.8532000, -6.0922627, -8.8532000, -6.0922627, -1.6223526, 1.6223526
1: -14.2740726, -11.7365370, -14.2740726, -11.7365370, -1.8242674, 1.8242671
2: -8.4603796, -6.5364571, -8.4603796, -6.5364571, -1.3472347, 1.3472347
3: -10.8423510, -8.4118052, -10.8423510, -8.4118052, -2.1507540, 2.1507545
4: -4.4535594, -2.5585070, -4.4535594, -2.5585070, -1.6094904, 1.6094904
5: -2.9140515, -1.0502224, -2.9140515, -1.0502224, -1.4807110, 1.4807110
6: -10.6835003, -7.8720369, -10.6835003, -7.8720369, -1.7177634, 1.7177637
7: -1.8835671, 0.1428504, -1.8835671, 0.1428504, -1.8345613, 1.8345613
8: -4.4770365, -2.3309865, -4.4770365, -2.3309865, -1.4947309, 1.4947309
9: 1.3801057, 2.6901963, 1.3801057, 2.6901963, -1.0198054, 1.0198057

Time for backsubstitution: 21.29 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80
type: B, layer: 1, pos: 80

Time for candidate selection: 0.27 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5989027, upper bound: 0.5943334
time: 4.08 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5987360, upper bound: 0.5984297
time: 4.62 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 30.28 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 30.28
Output dim: 9, lower bound: -0.5985977, upper bound: 0.5948069
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 30.28
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5988801
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.28
Output dim: 9, lower bound: -0.5985977, upper bound: 0.5948781
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.28
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5989485
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 30.28
Output dim: 9, lower bound: -0.5946488, upper bound: 0.5985988
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 30.28
Output dim: 9, lower bound: -0.5987360, upper bound: 0.5984304
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 30.28
Output dim: 9, lower bound: -0.5989027, upper bound: 0.5943334
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 30.28
Output dim: 9, lower bound: -0.5987360, upper bound: 0.5984297

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -8.8529758, -6.0920954, -8.8529778, -6.0920916, -1.6221848, 1.6221786
1: -14.2732496, -11.7366638, -14.2732525, -11.7366600, -1.8241048, 1.8241057
2: -8.4603920, -6.5364165, -8.4603920, -6.5364156, -1.3472128, 1.3472130
3: -10.8424587, -8.4108448, -10.8424606, -8.4108410, -2.1505651, 2.1505728
4: -4.4520473, -2.5586717, -4.4520507, -2.5586698, -1.6091557, 1.6091578
5: -2.9136362, -1.0502634, -2.9136343, -1.0502625, -1.4807291, 1.4807284
6: -10.6823845, -7.8721642, -10.6823807, -7.8721628, -1.7176251, 1.7176197
7: -1.8836939, 0.1442804, -1.8836927, 0.1442800, -1.8347425, 1.8347440
8: -4.4790015, -2.3308239, -4.4790001, -2.3308208, -1.4949741, 1.4949732
9: 1.3788505, 2.6902032, 1.3788490, 2.6902046, -1.0200310, 1.0200317

Time for backsubstitution: 21.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 758
type: B, layer: 3, pos: 758
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 1452
type: B, layer: 3, pos: 1452
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: B, layer: 3, pos: 556
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 2825
type: B, layer: 3, pos: 2825
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 1684
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 2544
type: B, layer: 3, pos: 2544
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 563
type: A, layer: 3, pos: 563
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 2152
type: A, layer: 3, pos: 2152
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 598
type: A, layer: 3, pos: 598
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 907
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2909
type: B, layer: 3, pos: 2525
type: A, layer: 3, pos: 2525
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1692
type: A, layer: 3, pos: 1692
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 422
type: A, layer: 3, pos: 422
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 1383
type: B, layer: 3, pos: 1383
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: B, layer: 3, pos: 1104
type: A, layer: 3, pos: 1104
type: B, layer: 3, pos: 2518
type: A, layer: 3, pos: 2518
type: A, layer: 3, pos: 920
type: B, layer: 3, pos: 920
type: B, layer: 3, pos: 1492
type: A, layer: 3, pos: 1492
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1985
type: B, layer: 3, pos: 1985
type: A, layer: 3, pos: 897
type: B, layer: 3, pos: 897
type: A, layer: 3, pos: 173
type: B, layer: 3, pos: 173
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: B, layer: 3, pos: 904
type: A, layer: 3, pos: 904

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 1942

## Relational analysis of NS_A1_B1_A1_A1

### Relational analysis result of NS_A1_B1_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5821296, upper bound: 0.5863864
time: 3.89 seconds

## Relational analysis of NS_A1_B1_A1_A2

### Relational analysis result of NS_A1_B1_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5905704, upper bound: 0.5863879
time: 3.77 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -8.8529778, -6.0920897, -8.8529749, -6.0920906, -1.6221805, 1.6221871
1: -14.2732544, -11.7366562, -14.2732534, -11.7366552, -1.8241072, 1.8241065
2: -8.4603930, -6.5364137, -8.4603930, -6.5364146, -1.3472147, 1.3472152
3: -10.8424568, -8.4108410, -10.8424597, -8.4108410, -2.1505761, 2.1505666
4: -4.4520521, -2.5586674, -4.4520526, -2.5586677, -1.6091595, 1.6091592
5: -2.9136353, -1.0502625, -2.9136348, -1.0502625, -1.4807301, 1.4807303
6: -10.6823807, -7.8721638, -10.6823797, -7.8721619, -1.7176194, 1.7176249
7: -1.8836951, 0.1442804, -1.8836935, 0.1442809, -1.8347454, 1.8347440
8: -4.4789991, -2.3308210, -4.4790001, -2.3308203, -1.4949789, 1.4949760
9: 1.3788483, 2.6902061, 1.3788486, 2.6902051, -1.0200336, 1.0200325

Time for backsubstitution: 21.13 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 758
type: B, layer: 3, pos: 758
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2228
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1452
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: B, layer: 3, pos: 556
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 2825
type: A, layer: 3, pos: 2825
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 1684
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 2544
type: B, layer: 3, pos: 2544
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 563
type: A, layer: 3, pos: 563
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 1143
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 2152
type: A, layer: 3, pos: 2152
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 598
type: A, layer: 3, pos: 598
type: B, layer: 3, pos: 907
type: A, layer: 3, pos: 907
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2909
type: B, layer: 3, pos: 2525
type: A, layer: 3, pos: 2525
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 1692
type: A, layer: 3, pos: 1692
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 422
type: B, layer: 3, pos: 422
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 1383
type: A, layer: 3, pos: 1383
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1104
type: A, layer: 3, pos: 1104
type: A, layer: 3, pos: 2518
type: B, layer: 3, pos: 2518
type: A, layer: 3, pos: 920
type: B, layer: 3, pos: 920
type: B, layer: 3, pos: 1492
type: A, layer: 3, pos: 1492
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1985
type: A, layer: 3, pos: 1985
type: A, layer: 3, pos: 897
type: B, layer: 3, pos: 897
type: A, layer: 3, pos: 173
type: B, layer: 3, pos: 173
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 904
type: A, layer: 3, pos: 904

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 1942

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5819558, upper bound: 0.5903619
time: 3.71 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5903625, upper bound: 0.5903620
time: 4.24 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -8.8529758, -6.0920954, -8.8532009, -6.0922680, -1.6220140, 1.6206160
1: -14.2732496, -11.7366638, -14.2740726, -11.7365398, -1.8242249, 1.8249378
2: -8.4603920, -6.5364165, -8.4603786, -6.5364571, -1.3471680, 1.3472011
3: -10.8424587, -8.4108448, -10.8423519, -8.4118061, -2.1495976, 2.1504641
4: -4.4520473, -2.5586717, -4.4535570, -2.5585096, -1.6093183, 1.6106942
5: -2.9136362, -1.0502634, -2.9140501, -1.0502214, -1.4807692, 1.4811358
6: -10.6823845, -7.8721642, -10.6835022, -7.8720388, -1.7177444, 1.7187326
7: -1.8836939, 0.1442804, -1.8835676, 0.1428514, -1.8349257, 1.8349185
8: -4.4790015, -2.3308239, -4.4770355, -2.3309886, -1.4970117, 1.4950483
9: 1.3788505, 2.6902032, 1.3801069, 2.6901939, -1.0212464, 1.0201709

Time for backsubstitution: 21.67 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 758
type: B, layer: 3, pos: 758
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 1452
type: B, layer: 3, pos: 1452
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 556
type: A, layer: 3, pos: 556
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 2825
type: B, layer: 3, pos: 2825
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 1684
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 2544
type: B, layer: 3, pos: 2544
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 563
type: A, layer: 3, pos: 563
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 2152
type: A, layer: 3, pos: 2152
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 598
type: A, layer: 3, pos: 598
type: B, layer: 3, pos: 907
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 2909
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2525
type: A, layer: 3, pos: 2525
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1692
type: A, layer: 3, pos: 1692
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 422
type: A, layer: 3, pos: 422
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 1503
type: A, layer: 3, pos: 1383
type: B, layer: 3, pos: 1383
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: B, layer: 3, pos: 1104
type: A, layer: 3, pos: 1104
type: B, layer: 3, pos: 2518
type: A, layer: 3, pos: 2518
type: A, layer: 3, pos: 920
type: B, layer: 3, pos: 920
type: B, layer: 3, pos: 1492
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1492
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1985
type: B, layer: 3, pos: 1985
type: A, layer: 3, pos: 897
type: B, layer: 3, pos: 897
type: A, layer: 3, pos: 173
type: B, layer: 3, pos: 173
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 904
type: B, layer: 3, pos: 904

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 1942

## Relational analysis of NS_A1_B2_A1_A1

### Relational analysis result of NS_A1_B2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5816088, upper bound: 0.5864445
time: 3.76 seconds

## Relational analysis of NS_A1_B2_A1_A2

### Relational analysis result of NS_A1_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5900517, upper bound: 0.5864433
time: 3.94 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -8.8529778, -6.0920897, -8.8532000, -6.0922632, -1.6220102, 1.6206245
1: -14.2732544, -11.7366562, -14.2740726, -11.7365341, -1.8242269, 1.8249388
2: -8.4603930, -6.5364137, -8.4603786, -6.5364566, -1.3471694, 1.3472033
3: -10.8424568, -8.4108410, -10.8423510, -8.4118061, -2.1496096, 2.1504564
4: -4.4520521, -2.5586674, -4.4535580, -2.5585063, -1.6093221, 1.6106956
5: -2.9136353, -1.0502625, -2.9140501, -1.0502224, -1.4807706, 1.4811380
6: -10.6823807, -7.8721638, -10.6835051, -7.8720369, -1.7177401, 1.7187386
7: -1.8836951, 0.1442804, -1.8835673, 0.1428504, -1.8349266, 1.8349195
8: -4.4789991, -2.3308210, -4.4770374, -2.3309841, -1.4970169, 1.4950521
9: 1.3788483, 2.6902061, 1.3801055, 2.6901956, -1.0212483, 1.0201715

Time for backsubstitution: 21.05 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 758
type: B, layer: 3, pos: 758
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 1452
type: B, layer: 3, pos: 1452
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 556
type: A, layer: 3, pos: 556
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 2825
type: B, layer: 3, pos: 2825
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 1684
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 2544
type: B, layer: 3, pos: 2544
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 563
type: A, layer: 3, pos: 563
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 2152
type: A, layer: 3, pos: 2152
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 598
type: A, layer: 3, pos: 598
type: B, layer: 3, pos: 907
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 2909
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2525
type: A, layer: 3, pos: 2525
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 46
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1692
type: A, layer: 3, pos: 1692
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 422
type: A, layer: 3, pos: 422
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 1503
type: A, layer: 3, pos: 1383
type: B, layer: 3, pos: 1383
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: B, layer: 3, pos: 1104
type: A, layer: 3, pos: 1104
type: B, layer: 3, pos: 2518
type: A, layer: 3, pos: 2518
type: A, layer: 3, pos: 920
type: B, layer: 3, pos: 920
type: B, layer: 3, pos: 1492
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1492
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1985
type: B, layer: 3, pos: 1985
type: A, layer: 3, pos: 897
type: B, layer: 3, pos: 897
type: A, layer: 3, pos: 173
type: B, layer: 3, pos: 173
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 904
type: B, layer: 3, pos: 904

Time for candidate selection: 0.53 seconds

### Candidate
type: A, layer: 3, pos: 1942

## Relational analysis of NS_A1_B2_A2_A1

### Relational analysis result of NS_A1_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5814349, upper bound: 0.5904180
time: 3.66 seconds

## Relational analysis of NS_A1_B2_A2_A2

### Relational analysis result of NS_A1_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5898199, upper bound: 0.5904179
time: 8.03 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -8.8532009, -6.0922680, -8.8529758, -6.0920954, -1.6206155, 1.6220140
1: -14.2740726, -11.7365398, -14.2732496, -11.7366638, -1.8249378, 1.8242249
2: -8.4603786, -6.5364571, -8.4603920, -6.5364165, -1.3472009, 1.3471677
3: -10.8423519, -8.4118061, -10.8424587, -8.4108448, -2.1504641, 2.1495976
4: -4.4535570, -2.5585096, -4.4520473, -2.5586717, -1.6106944, 1.6093183
5: -2.9140501, -1.0502214, -2.9136362, -1.0502634, -1.4811358, 1.4807692
6: -10.6835022, -7.8720388, -10.6823845, -7.8721642, -1.7187328, 1.7177444
7: -1.8835676, 0.1428514, -1.8836939, 0.1442804, -1.8349180, 1.8349257
8: -4.4770355, -2.3309886, -4.4790015, -2.3308239, -1.4950480, 1.4970114
9: 1.3801069, 2.6901939, 1.3788505, 2.6902032, -1.0201709, 1.0212464

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 758
type: A, layer: 3, pos: 758
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: B, layer: 3, pos: 556
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 2825
type: A, layer: 3, pos: 2825
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1684
type: A, layer: 3, pos: 1684
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 2544
type: A, layer: 3, pos: 2544
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 563
type: B, layer: 3, pos: 563
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 2152
type: B, layer: 3, pos: 2152
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 598
type: B, layer: 3, pos: 598
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 907
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2909
type: A, layer: 3, pos: 2525
type: B, layer: 3, pos: 2525
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 46
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 1692
type: B, layer: 3, pos: 1692
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 422
type: B, layer: 3, pos: 422
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 1503
type: B, layer: 3, pos: 1383
type: A, layer: 3, pos: 1383
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: A, layer: 3, pos: 1104
type: B, layer: 3, pos: 1104
type: A, layer: 3, pos: 2518
type: B, layer: 3, pos: 2518
type: B, layer: 3, pos: 920
type: A, layer: 3, pos: 920
type: A, layer: 3, pos: 1492
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1492
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1985
type: A, layer: 3, pos: 1985
type: B, layer: 3, pos: 897
type: A, layer: 3, pos: 897
type: B, layer: 3, pos: 173
type: A, layer: 3, pos: 173
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 904
type: A, layer: 3, pos: 904

Time for candidate selection: 0.44 seconds

### Candidate
type: B, layer: 3, pos: 1942

## Relational analysis of NS_A2_B1_B1_B1

### Relational analysis result of NS_A2_B1_B1_B1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5864435, upper bound: 0.5816096
time: 3.59 seconds

## Relational analysis of NS_A2_B1_B1_B2

### Relational analysis result of NS_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5864435, upper bound: 0.5900515
time: 3.73 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -8.8532000, -6.0922632, -8.8529778, -6.0920897, -1.6206245, 1.6220107
1: -14.2740726, -11.7365341, -14.2732544, -11.7366562, -1.8249388, 1.8242271
2: -8.4603786, -6.5364566, -8.4603930, -6.5364137, -1.3472033, 1.3471699
3: -10.8423510, -8.4118061, -10.8424568, -8.4108410, -2.1504564, 2.1496096
4: -4.4535580, -2.5585063, -4.4520521, -2.5586674, -1.6106958, 1.6093221
5: -2.9140501, -1.0502224, -2.9136353, -1.0502625, -1.4811382, 1.4807703
6: -10.6835051, -7.8720369, -10.6823807, -7.8721638, -1.7187386, 1.7177398
7: -1.8835673, 0.1428504, -1.8836951, 0.1442804, -1.8349199, 1.8349266
8: -4.4770374, -2.3309841, -4.4789991, -2.3308210, -1.4950523, 1.4970169
9: 1.3801055, 2.6901956, 1.3788483, 2.6902061, -1.0201716, 1.0212482

Time for backsubstitution: 21.92 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 758
type: A, layer: 3, pos: 758
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: B, layer: 3, pos: 556
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 1494
type: B, layer: 3, pos: 2825
type: A, layer: 3, pos: 2825
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1684
type: A, layer: 3, pos: 1684
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 1151
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 2544
type: A, layer: 3, pos: 2544
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 563
type: B, layer: 3, pos: 563
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 2152
type: B, layer: 3, pos: 2152
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 598
type: B, layer: 3, pos: 598
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 907
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2909
type: A, layer: 3, pos: 2525
type: B, layer: 3, pos: 2525
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 46
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 1692
type: B, layer: 3, pos: 1692
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 422
type: B, layer: 3, pos: 422
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 1503
type: B, layer: 3, pos: 1383
type: A, layer: 3, pos: 1383
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: A, layer: 3, pos: 1104
type: B, layer: 3, pos: 1104
type: A, layer: 3, pos: 2518
type: B, layer: 3, pos: 2518
type: B, layer: 3, pos: 920
type: A, layer: 3, pos: 920
type: A, layer: 3, pos: 1492
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1492
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 1985
type: A, layer: 3, pos: 1985
type: B, layer: 3, pos: 897
type: A, layer: 3, pos: 897
type: B, layer: 3, pos: 173
type: A, layer: 3, pos: 173
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 904
type: A, layer: 3, pos: 904

Time for candidate selection: 0.45 seconds

### Candidate
type: B, layer: 3, pos: 1942

## Relational analysis of NS_A2_B1_B2_B1

### Relational analysis result of NS_A2_B1_B2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5904187, upper bound: 0.5814357
time: 3.80 seconds

## Relational analysis of NS_A2_B1_B2_B2

### Relational analysis result of NS_A2_B1_B2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5904187, upper bound: 0.5898193
time: 3.63 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -8.8531990, -6.0922699, -8.8532009, -6.0922680, -1.6223569, 1.6223507
1: -14.2740707, -11.7365427, -14.2740726, -11.7365398, -1.8242636, 1.8242662
2: -8.4603786, -6.5364590, -8.4603786, -6.5364571, -1.3472323, 1.3472319
3: -10.8423500, -8.4118099, -10.8423519, -8.4118061, -2.1507521, 2.1507592
4: -4.4535561, -2.5585117, -4.4535570, -2.5585096, -1.6094866, 1.6094880
5: -2.9140515, -1.0502224, -2.9140501, -1.0502214, -1.4807105, 1.4807093
6: -10.6835051, -7.8720369, -10.6835022, -7.8720388, -1.7177668, 1.7177610
7: -1.8835666, 0.1428509, -1.8835676, 0.1428514, -1.8345613, 1.8345623
8: -4.4770384, -2.3309896, -4.4770355, -2.3309886, -1.4947286, 1.4947278
9: 1.3801079, 2.6901937, 1.3801069, 2.6901939, -1.0198040, 1.0198047

Time for backsubstitution: 22.03 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 758
type: B, layer: 3, pos: 758
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 1452
type: B, layer: 3, pos: 1452
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: B, layer: 3, pos: 556
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 2825
type: B, layer: 3, pos: 2825
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 1684
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 2544
type: B, layer: 3, pos: 2544
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 563
type: A, layer: 3, pos: 563
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 2152
type: A, layer: 3, pos: 2152
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 598
type: A, layer: 3, pos: 598
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 907
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2909
type: B, layer: 3, pos: 2525
type: A, layer: 3, pos: 2525
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 46
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 1692
type: A, layer: 3, pos: 1692
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: B, layer: 3, pos: 422
type: A, layer: 3, pos: 422
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 1503
type: A, layer: 3, pos: 1383
type: B, layer: 3, pos: 1383
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: B, layer: 3, pos: 1104
type: A, layer: 3, pos: 1104
type: A, layer: 3, pos: 2518
type: B, layer: 3, pos: 2518
type: A, layer: 3, pos: 920
type: B, layer: 3, pos: 920
type: B, layer: 3, pos: 1492
type: A, layer: 3, pos: 1492
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: A, layer: 3, pos: 1985
type: B, layer: 3, pos: 1985
type: B, layer: 3, pos: 897
type: A, layer: 3, pos: 897
type: A, layer: 3, pos: 173
type: B, layer: 3, pos: 173
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 904
type: B, layer: 3, pos: 904

Time for candidate selection: 0.47 seconds

### Candidate
type: A, layer: 3, pos: 1942

## Relational analysis of NS_A2_B2_A1_A1

### Relational analysis result of NS_A2_B2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5819241, upper bound: 0.5858549
time: 3.85 seconds

## Relational analysis of NS_A2_B2_A1_A2

### Relational analysis result of NS_A2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5903676, upper bound: 0.5858549
time: 4.37 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -8.8532009, -6.0922627, -8.8532000, -6.0922632, -1.6223521, 1.6223588
1: -14.2740726, -11.7365360, -14.2740726, -11.7365341, -1.8242669, 1.8242664
2: -8.4603796, -6.5364566, -8.4603786, -6.5364566, -1.3472352, 1.3472347
3: -10.8423481, -8.4118071, -10.8423510, -8.4118061, -2.1507640, 2.1507530
4: -4.4535599, -2.5585060, -4.4535580, -2.5585063, -1.6094894, 1.6094890
5: -2.9140515, -1.0502234, -2.9140501, -1.0502224, -1.4807105, 1.4807127
6: -10.6835041, -7.8720360, -10.6835051, -7.8720369, -1.7177625, 1.7177670
7: -1.8835678, 0.1428514, -1.8835673, 0.1428504, -1.8345609, 1.8345613
8: -4.4770365, -2.3309872, -4.4770374, -2.3309841, -1.4947333, 1.4947312
9: 1.3801055, 2.6901960, 1.3801055, 2.6901956, -1.0198052, 1.0198052

Time for backsubstitution: 21.97 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1942
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 758
type: B, layer: 3, pos: 758
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 2228
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1452
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: B, layer: 3, pos: 556
type: A, layer: 3, pos: 2627
type: B, layer: 3, pos: 2627
type: B, layer: 3, pos: 1101
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 2825
type: A, layer: 3, pos: 2825
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 1684
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 2544
type: B, layer: 3, pos: 2544
type: A, layer: 3, pos: 1731
type: B, layer: 3, pos: 1731
type: B, layer: 3, pos: 563
type: A, layer: 3, pos: 563
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 1143
type: B, layer: 3, pos: 311
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 2152
type: A, layer: 3, pos: 2152
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 3104
type: B, layer: 3, pos: 598
type: A, layer: 3, pos: 598
type: B, layer: 3, pos: 907
type: A, layer: 3, pos: 907
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2909
type: B, layer: 3, pos: 2525
type: A, layer: 3, pos: 2525
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 913
type: A, layer: 3, pos: 46
type: B, layer: 3, pos: 46
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 1692
type: A, layer: 3, pos: 1692
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 422
type: B, layer: 3, pos: 422
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 1503
type: B, layer: 3, pos: 1383
type: A, layer: 3, pos: 1383
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1104
type: A, layer: 3, pos: 1104
type: A, layer: 3, pos: 2518
type: B, layer: 3, pos: 2518
type: A, layer: 3, pos: 920
type: B, layer: 3, pos: 920
type: B, layer: 3, pos: 1492
type: A, layer: 3, pos: 1492
type: A, layer: 3, pos: 158
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1985
type: A, layer: 3, pos: 1985
type: A, layer: 3, pos: 897
type: B, layer: 3, pos: 897
type: A, layer: 3, pos: 173
type: B, layer: 3, pos: 173
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 904
type: A, layer: 3, pos: 904

Time for candidate selection: 0.51 seconds

### Candidate
type: A, layer: 3, pos: 1942

## Relational analysis of NS_A2_B2_A2_A1

### Relational analysis result of NS_A2_B2_A2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5817507, upper bound: 0.5898196
time: 3.78 seconds

## Relational analysis of NS_A2_B2_A2_A2

### Relational analysis result of NS_A2_B2_A2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5901373, upper bound: 0.5898192
time: 4.24 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 30.52 seconds
NS_A1_B1_A1_A1, status: Status.VERIFIED, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5821296, upper bound: 0.5863864
NS_A1_B1_A1_A2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5905704, upper bound: 0.5863879
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5819558, upper bound: 0.5903619
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5903625, upper bound: 0.5903620
NS_A1_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5816088, upper bound: 0.5864445
NS_A1_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5900517, upper bound: 0.5864433
NS_A1_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5814349, upper bound: 0.5904180
NS_A1_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5898199, upper bound: 0.5904179
NS_A2_B1_B1_B1, status: Status.VERIFIED, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5864435, upper bound: 0.5816096
NS_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5864435, upper bound: 0.5900515
NS_A2_B1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5904187, upper bound: 0.5814357
NS_A2_B1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5904187, upper bound: 0.5898193
NS_A2_B2_A1_A1, status: Status.VERIFIED, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5819241, upper bound: 0.5858549
NS_A2_B2_A1_A2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5903676, upper bound: 0.5858549
NS_A2_B2_A2_A1, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5817507, upper bound: 0.5898196
NS_A2_B2_A2_A2, status: Status.UNKNOWN, split count: 4, time: 30.52
Output dim: 9, lower bound: -0.5901373, upper bound: 0.5898192

## BFS NS instance: NS_A1_B1_A1_A2

### Backsubstitution after applying NS history:
0: -8.8711910, -6.1618371, -8.8509445, -6.1329751, -1.6377282, 1.5937438
1: -14.2726879, -11.7461443, -14.2695522, -11.7425213, -1.8068905, 1.8111446
2: -8.4630051, -6.5441566, -8.4598913, -6.5433183, -1.3457532, 1.3403571
3: -10.8387394, -8.4099703, -10.8387299, -8.4122581, -2.1475358, 2.1430035
4: -4.4340620, -2.5459352, -4.4394426, -2.5591333, -1.5935473, 1.6109107
5: -2.9300532, -1.1013889, -2.9118419, -1.0779982, -1.4820747, 1.4715023
6: -10.7060699, -7.9026027, -10.6779938, -7.8910494, -1.7069573, 1.7007713
7: -1.8667958, 0.2058287, -1.8692765, 0.1442518, -1.8151636, 1.8533010
8: -4.5690660, -2.3828614, -4.4789686, -2.3616204, -1.5616126, 1.4488826
9: 1.4056544, 2.7507749, 1.4007986, 2.6897073, -1.0039876, 1.0797553

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 758
type: A, layer: 3, pos: 758
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 1452
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: B, layer: 3, pos: 556
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 2825
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2825
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1684
type: A, layer: 3, pos: 563
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 2544
type: A, layer: 3, pos: 2544
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 1143
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2152
type: A, layer: 3, pos: 2152
type: B, layer: 3, pos: 2525
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 907
type: A, layer: 3, pos: 598
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 2909
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 46
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 2525
type: A, layer: 3, pos: 1692
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 563
type: B, layer: 3, pos: 422
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 1503
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 1692
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 422
type: A, layer: 3, pos: 1788
type: A, layer: 3, pos: 1383
type: B, layer: 3, pos: 1383
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1104
type: B, layer: 3, pos: 1104
type: B, layer: 3, pos: 2518
type: A, layer: 3, pos: 2518
type: B, layer: 3, pos: 920
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1492
type: A, layer: 3, pos: 1492
type: B, layer: 3, pos: 1985
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1985
type: A, layer: 3, pos: 920
type: B, layer: 3, pos: 897
type: A, layer: 3, pos: 897
type: A, layer: 3, pos: 173
type: B, layer: 3, pos: 173
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 904
type: B, layer: 3, pos: 904

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_B1_A1_A2_A1

### Relational analysis result of NS_A1_B1_A1_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5837440, upper bound: 0.5780428
time: 3.92 seconds

## Relational analysis of NS_A1_B1_A1_A2_A2

### Relational analysis result of NS_A1_B1_A1_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5829187, upper bound: 0.5787122
time: 4.53 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -8.8516207, -6.1161847, -8.8523331, -6.1035485, -1.6038485, 1.5921392
1: -14.2699986, -11.7420158, -14.2717733, -11.7389545, -1.8111153, 1.8129227
2: -8.4601250, -6.5440693, -8.4602633, -6.5397615, -1.3429813, 1.3383145
3: -10.8395433, -8.4121456, -10.8411961, -8.4114504, -2.1443143, 2.1445956
4: -4.4408212, -2.5592458, -4.4471502, -2.5589206, -1.5984979, 1.6034663
5: -2.9110975, -1.0670958, -2.9124937, -1.0574970, -1.4643035, 1.4595864
6: -10.6773453, -7.8930368, -10.6801453, -7.8811464, -1.6923089, 1.6933870
7: -1.8507466, 0.1442289, -1.8694756, 0.1442580, -1.8072691, 1.8226886
8: -4.4789805, -2.3849349, -4.4789915, -2.3549387, -1.4712420, 1.4474344
9: 1.4074531, 2.6896722, 1.3921309, 2.6899705, -0.9961014, 1.0056443

Time for backsubstitution: 21.93 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: A, layer: 3, pos: 758
type: B, layer: 3, pos: 758
type: B, layer: 3, pos: 1942
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 1452
type: B, layer: 3, pos: 1452
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: B, layer: 3, pos: 556
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 2460
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 2825
type: B, layer: 3, pos: 2825
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 921
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1684
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 765
type: B, layer: 3, pos: 2544
type: A, layer: 3, pos: 2544
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 563
type: A, layer: 3, pos: 1143
type: B, layer: 3, pos: 1143
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 563
type: B, layer: 3, pos: 2152
type: A, layer: 3, pos: 2152
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 598
type: A, layer: 3, pos: 598
type: B, layer: 3, pos: 907
type: B, layer: 3, pos: 2525
type: B, layer: 3, pos: 2909
type: A, layer: 3, pos: 2909
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 2525
type: A, layer: 3, pos: 1843
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 46
type: A, layer: 3, pos: 913
type: B, layer: 3, pos: 2607
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 1692
type: B, layer: 3, pos: 1692
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 422
type: A, layer: 3, pos: 1503
type: A, layer: 3, pos: 422
type: B, layer: 3, pos: 1503
type: B, layer: 3, pos: 1383
type: A, layer: 3, pos: 1383
type: A, layer: 3, pos: 1788
type: B, layer: 3, pos: 1788
type: B, layer: 3, pos: 1104
type: A, layer: 3, pos: 1104
type: B, layer: 3, pos: 2518
type: A, layer: 3, pos: 2518
type: B, layer: 3, pos: 920
type: A, layer: 3, pos: 920
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1492
type: A, layer: 3, pos: 1492
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1985
type: B, layer: 3, pos: 1985
type: A, layer: 3, pos: 897
type: B, layer: 3, pos: 897
type: A, layer: 3, pos: 173
type: B, layer: 3, pos: 173
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 904
type: B, layer: 3, pos: 904

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_B1_A2_A1_A1

### Relational analysis result of NS_A1_B1_A2_A1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5759809, upper bound: 0.5819654
time: 3.90 seconds

## Relational analysis of NS_A1_B1_A2_A1_A2

### Relational analysis result of NS_A1_B1_A2_A1_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5731797, upper bound: 0.5826694
time: 4.30 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -8.8711910, -6.1618285, -8.8509474, -6.1329694, -1.6377239, 1.5937529
1: -14.2726898, -11.7461338, -14.2695551, -11.7425184, -1.8068914, 1.8111455
2: -8.4630041, -6.5441551, -8.4598923, -6.5433168, -1.3457556, 1.3403594
3: -10.8387375, -8.4099684, -10.8387308, -8.4122581, -2.1475453, 2.1429939
4: -4.4340658, -2.5459313, -4.4394450, -2.5591304, -1.5935497, 1.6109114
5: -2.9300518, -1.1013880, -2.9118433, -1.0779972, -1.4820743, 1.4715052
6: -10.7060652, -7.9026051, -10.6779966, -7.8910475, -1.7069535, 1.7007773
7: -1.8667951, 0.2058296, -1.8692765, 0.1442513, -1.8151646, 1.8533010
8: -4.5690632, -2.3828568, -4.4789691, -2.3616180, -1.5616169, 1.4488854
9: 1.4056525, 2.7507765, 1.4007988, 2.6897094, -1.0039885, 1.0797553

Time for backsubstitution: 22.13 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 766
type: B, layer: 3, pos: 766
type: A, layer: 3, pos: 1683
type: B, layer: 3, pos: 1683
type: B, layer: 3, pos: 758
type: A, layer: 3, pos: 758
type: B, layer: 3, pos: 1109
type: A, layer: 3, pos: 1109
type: B, layer: 3, pos: 2228
type: A, layer: 3, pos: 2228
type: B, layer: 3, pos: 1942
type: A, layer: 3, pos: 1452
type: B, layer: 3, pos: 1452
type: A, layer: 3, pos: 1682
type: B, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: B, layer: 3, pos: 556
type: B, layer: 3, pos: 2627
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1101
type: B, layer: 3, pos: 1101
type: B, layer: 3, pos: 2382
type: A, layer: 3, pos: 2382
type: B, layer: 3, pos: 2460
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 1494
type: B, layer: 3, pos: 1494
type: A, layer: 3, pos: 2825
type: A, layer: 3, pos: 1684
type: B, layer: 3, pos: 921
type: B, layer: 3, pos: 2825
type: A, layer: 3, pos: 921
type: B, layer: 3, pos: 1684
type: A, layer: 3, pos: 563
type: B, layer: 3, pos: 1151
type: A, layer: 3, pos: 1151
type: B, layer: 3, pos: 765
type: A, layer: 3, pos: 765
type: B, layer: 3, pos: 2544
type: A, layer: 3, pos: 2544
type: B, layer: 3, pos: 1731
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 311
type: B, layer: 3, pos: 1143
type: B, layer: 3, pos: 311
type: B, layer: 3, pos: 2152
type: A, layer: 3, pos: 2152
type: B, layer: 3, pos: 2525
type: B, layer: 3, pos: 3104
type: A, layer: 3, pos: 3104
type: B, layer: 3, pos: 907
type: A, layer: 3, pos: 598
type: B, layer: 3, pos: 598
type: B, layer: 3, pos: 2909
type: A, layer: 3, pos: 2909
type: B, layer: 3, pos: 2607
type: B, layer: 3, pos: 1843
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 907
type: B, layer: 3, pos: 913
type: B, layer: 3, pos: 46
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 1920
type: B, layer: 3, pos: 1920
type: A, layer: 3, pos: 2525
type: A, layer: 3, pos: 1692
type: A, layer: 3, pos: 2607
type: B, layer: 3, pos: 563
type: B, layer: 3, pos: 422
type: B, layer: 3, pos: 2145
type: A, layer: 3, pos: 1503
type: A, layer: 3, pos: 2145
type: B, layer: 3, pos: 1692
type: B, layer: 3, pos: 1503
type: A, layer: 3, pos: 422
type: A, layer: 3, pos: 1788
type: A, layer: 3, pos: 1383
type: B, layer: 3, pos: 1383
type: B, layer: 3, pos: 1788
type: A, layer: 3, pos: 1104
type: B, layer: 3, pos: 1104
type: B, layer: 3, pos: 2518
type: A, layer: 3, pos: 2518
type: B, layer: 3, pos: 920
type: B, layer: 3, pos: 158
type: B, layer: 3, pos: 1492
type: A, layer: 3, pos: 1492
type: B, layer: 3, pos: 1985
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1985
type: A, layer: 3, pos: 920
type: B, layer: 3, pos: 897
type: A, layer: 3, pos: 897
type: A, layer: 3, pos: 173
type: B, layer: 3, pos: 173
type: A, layer: 3, pos: 1773
type: B, layer: 3, pos: 1773
type: A, layer: 3, pos: 904
type: B, layer: 3, pos: 904

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 766

## Relational analysis of NS_A1_B1_A2_A2_A1

### Relational analysis result of NS_A1_B1_A2_A2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5834936, upper bound: 0.5819651
time: 4.51 seconds

## Relational analysis of NS_A1_B1_A2_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2_A2
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5826700, upper bound: 0.5826696
time: 4.31 seconds

## BFS NS instance: NS_A1_B2_A1_A2

### Backsubstitution after applying NS history:
0: -8.8711910, -6.1618371, -8.8511686, -6.1331491, -1.6375570, 1.5921812
1: -14.2726879, -11.7461443, -14.2703733, -11.7424021, -1.8070107, 1.8119760
2: -8.4630051, -6.5441566, -8.4598770, -6.5433602, -1.3457084, 1.3403451
3: -10.8387394, -8.4099703, -10.8386230, -8.4132233, -2.1465697, 2.1428919
4: -4.4340620, -2.5459352, -4.4409547, -2.5589724, -1.5937076, 1.6124446
5: -2.9300532, -1.1013889, -2.9122577, -1.0779572, -1.4821157, 1.4719098
6: -10.7060699, -7.9026027, -10.6791172, -7.8909225, -1.7070765, 1.7018838
7: -1.8667958, 0.2058287, -1.8691490, 0.1428218, -1.8153458, 1.8535256
8: -4.5690660, -2.3828614, -4.4770041, -2.3617876, -1.5636501, 1.4489584
9: 1.4056544, 2.7507749, 1.4020574, 2.6896977, -1.0052025, 1.0798938

Time for backsubstitution: 22.02 seconds

Time for candidate selection: 0.00 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 56.63 + 545.10 = 601.73 seconds
