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
execution time: IAR + RelationalAnalysis = 24.60 + 34.95 = 59.55 seconds
status: Status.UNKNOWN
relational distance
Output dim: 9, lower bound: -0.5992169, upper bound: 0.5992170

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 66
type: A, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 66

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5991908
time: 4.06 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5989774, upper bound: 0.5989777
time: 4.03 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 8.40 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 8.40
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5991908
NS_A2, status: Status.UNKNOWN, split count: 1, time: 8.40
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

Time for backsubstitution: 21.58 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5986718
time: 4.40 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5989777
time: 4.16 seconds

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

Time for backsubstitution: 21.89 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 66
type: B, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 66

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5989774, upper bound: 0.5986723
time: 4.56 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5989774, upper bound: 0.5989776
time: 3.98 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 30.73 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 30.73
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5986718
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 30.73
Output dim: 9, lower bound: -0.5986723, upper bound: 0.5989777
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 30.73
Output dim: 9, lower bound: -0.5989774, upper bound: 0.5986723
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 30.73
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

Time for backsubstitution: 22.68 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5985977, upper bound: 0.5948069
time: 4.05 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5988801
time: 4.31 seconds

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

Time for backsubstitution: 22.80 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5985977, upper bound: 0.5948781
time: 4.05 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5989485
time: 4.68 seconds

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

Time for backsubstitution: 22.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80

Time for candidate selection: 0.28 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5989027, upper bound: 0.5943332
time: 3.97 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5987360, upper bound: 0.5984298
time: 5.02 seconds

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

Time for backsubstitution: 22.76 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: A, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5989027, upper bound: 0.5943334
time: 4.37 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5987360, upper bound: 0.5984297
time: 4.81 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 32.24 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.24
Output dim: 9, lower bound: -0.5985977, upper bound: 0.5948069
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.24
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5988801
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.24
Output dim: 9, lower bound: -0.5985977, upper bound: 0.5948781
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.24
Output dim: 9, lower bound: -0.5984303, upper bound: 0.5989485
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 32.24
Output dim: 9, lower bound: -0.5989027, upper bound: 0.5943332
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 32.24
Output dim: 9, lower bound: -0.5987360, upper bound: 0.5984298
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 32.24
Output dim: 9, lower bound: -0.5989027, upper bound: 0.5943334
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 32.24
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

Time for backsubstitution: 22.05 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5948053, upper bound: 0.5948052
time: 3.70 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5948053, upper bound: 0.5948047
time: 5.61 seconds

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

Time for backsubstitution: 22.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5948053, upper bound: 0.5988814
time: 3.70 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5948053, upper bound: 0.5988797
time: 4.99 seconds

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

Time for backsubstitution: 22.15 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5943331, upper bound: 0.5948762
time: 3.80 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5943331, upper bound: 0.5948762
time: 4.63 seconds

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

Time for backsubstitution: 22.90 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5943331, upper bound: 0.5989501
time: 3.73 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5943331, upper bound: 0.5989507
time: 3.88 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -8.8531990, -6.0922699, -8.8529778, -6.0920916, -1.6206222, 1.6220083
1: -14.2740707, -11.7365427, -14.2732525, -11.7366600, -1.8249364, 1.8242266
2: -8.4603786, -6.5364590, -8.4603920, -6.5364156, -1.3471999, 1.3471677
3: -10.8423500, -8.4118099, -10.8424606, -8.4108410, -2.1504555, 2.1496043
4: -4.4535561, -2.5585117, -4.4520507, -2.5586698, -1.6106920, 1.6093194
5: -2.9140515, -1.0502224, -2.9136343, -1.0502625, -1.4811363, 1.4807692
6: -10.6835051, -7.8720369, -10.6823807, -7.8721628, -1.7187381, 1.7177389
7: -1.8835666, 0.1428509, -1.8836927, 0.1442800, -1.8349180, 1.8349257
8: -4.4770384, -2.3309896, -4.4790001, -2.3308208, -1.4950495, 1.4970107
9: 1.3801079, 2.6901937, 1.3788490, 2.6902046, -1.0201697, 1.0212475

Time for backsubstitution: 22.80 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80

Time for candidate selection: 0.30 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5948765, upper bound: 0.5943327
time: 3.89 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5948765, upper bound: 0.5943328
time: 4.29 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -8.8532009, -6.0922627, -8.8529749, -6.0920906, -1.6206179, 1.6220160
1: -14.2740726, -11.7365360, -14.2732534, -11.7366552, -1.8249388, 1.8242266
2: -8.4603796, -6.5364566, -8.4603930, -6.5364146, -1.3472033, 1.3471704
3: -10.8423481, -8.4118071, -10.8424597, -8.4108410, -2.1504664, 2.1495991
4: -4.4535599, -2.5585060, -4.4520526, -2.5586677, -1.6106958, 1.6093204
5: -2.9140515, -1.0502234, -2.9136348, -1.0502625, -1.4811382, 1.4807718
6: -10.6835041, -7.8720360, -10.6823797, -7.8721619, -1.7187333, 1.7177446
7: -1.8835678, 0.1428514, -1.8836935, 0.1442809, -1.8349190, 1.8349261
8: -4.4770365, -2.3309872, -4.4790001, -2.3308203, -1.4950547, 1.4970140
9: 1.3801055, 2.6901960, 1.3788486, 2.6902051, -1.0201719, 1.0212475

Time for backsubstitution: 22.82 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5948765, upper bound: 0.5984314
time: 3.70 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5948765, upper bound: 0.5984299
time: 4.34 seconds

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

Time for backsubstitution: 22.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80

Time for candidate selection: 0.28 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5946488, upper bound: 0.5943330
time: 3.61 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5946488, upper bound: 0.5943328
time: 5.45 seconds

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

Time for backsubstitution: 22.71 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 80

Time for candidate selection: 0.29 seconds

### Candidate
type: B, layer: 1, pos: 80

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5946488, upper bound: 0.5984314
time: 3.62 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5946488, upper bound: 0.5984299
time: 6.45 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 33.09 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5948053, upper bound: 0.5948052
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5948053, upper bound: 0.5948047
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5948053, upper bound: 0.5988814
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5948053, upper bound: 0.5988797
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5943331, upper bound: 0.5948762
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5943331, upper bound: 0.5948762
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5943331, upper bound: 0.5989501
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5943331, upper bound: 0.5989507
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5948765, upper bound: 0.5943327
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5948765, upper bound: 0.5943328
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5948765, upper bound: 0.5984314
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5948765, upper bound: 0.5984299
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5946488, upper bound: 0.5943330
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5946488, upper bound: 0.5943328
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5946488, upper bound: 0.5984314
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 33.09
Output dim: 9, lower bound: -0.5946488, upper bound: 0.5984299

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -8.8529758, -6.0920954, -8.8529758, -6.0920954, -1.6221838, 1.6221838
1: -14.2732496, -11.7366638, -14.2732496, -11.7366638, -1.8241038, 1.8241038
2: -8.4603920, -6.5364165, -8.4603920, -6.5364165, -1.3472133, 1.3472133
3: -10.8424587, -8.4108448, -10.8424587, -8.4108448, -2.1505737, 2.1505733
4: -4.4520473, -2.5586717, -4.4520473, -2.5586717, -1.6091557, 1.6091559
5: -2.9136362, -1.0502634, -2.9136362, -1.0502634, -1.4807286, 1.4807289
6: -10.6823845, -7.8721642, -10.6823845, -7.8721642, -1.7176251, 1.7176249
7: -1.8836939, 0.1442804, -1.8836939, 0.1442804, -1.8347445, 1.8347435
8: -4.4790015, -2.3308239, -4.4790015, -2.3308239, -1.4949737, 1.4949737
9: 1.3788505, 2.6902032, 1.3788505, 2.6902032, -1.0200315, 1.0200312

Time for backsubstitution: 22.80 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1942
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1684
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 2544
type: A, layer: 3, pos: 2525
type: A, layer: 3, pos: 2825
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 758
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2152
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 1383
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 422
type: A, layer: 3, pos: 2518
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 1788
type: A, layer: 3, pos: 2909
type: A, layer: 3, pos: 1692
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 920
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 563
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1503
type: A, layer: 3, pos: 1985
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 907
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 897
type: A, layer: 3, pos: 598
type: A, layer: 3, pos: 1104
type: A, layer: 3, pos: 173
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 1492
type: A, layer: 3, pos: 904

Time for candidate selection: 0.57 seconds

### Candidate
type: A, layer: 3, pos: 1942

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5821296, upper bound: 0.5863864
time: 4.11 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5905704, upper bound: 0.5863860
time: 4.38 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -8.8529758, -6.0920954, -8.8529778, -6.0920897, -1.6221843, 1.6221795
1: -14.2732496, -11.7366638, -14.2732544, -11.7366562, -1.8241029, 1.8241067
2: -8.4603920, -6.5364165, -8.4603930, -6.5364137, -1.3472147, 1.3472130
3: -10.8424587, -8.4108448, -10.8424568, -8.4108410, -2.1505661, 2.1505723
4: -4.4520473, -2.5586717, -4.4520521, -2.5586674, -1.6091547, 1.6091602
5: -2.9136362, -1.0502634, -2.9136353, -1.0502625, -1.4807291, 1.4807296
6: -10.6823845, -7.8721642, -10.6823807, -7.8721638, -1.7176232, 1.7176206
7: -1.8836939, 0.1442804, -1.8836951, 0.1442804, -1.8347435, 1.8347454
8: -4.4790015, -2.3308239, -4.4789991, -2.3308210, -1.4949775, 1.4949732
9: 1.3788505, 2.6902032, 1.3788483, 2.6902061, -1.0200295, 1.0200336

Time for backsubstitution: 22.80 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1942
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1684
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 2544
type: A, layer: 3, pos: 2525
type: A, layer: 3, pos: 2825
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 758
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2152
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 1383
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 422
type: A, layer: 3, pos: 2518
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 1788
type: A, layer: 3, pos: 2909
type: A, layer: 3, pos: 1692
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 920
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 563
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1503
type: A, layer: 3, pos: 1985
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 907
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 897
type: A, layer: 3, pos: 598
type: A, layer: 3, pos: 1104
type: A, layer: 3, pos: 173
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 1492
type: A, layer: 3, pos: 904

Time for candidate selection: 0.57 seconds

### Candidate
type: A, layer: 3, pos: 1942

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.VERIFIED
Output dim: 9, lower bound: -0.5821296, upper bound: 0.5863864
time: 4.04 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5905704, upper bound: 0.5863879
time: 3.78 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -8.8529778, -6.0920897, -8.8529758, -6.0920954, -1.6221795, 1.6221843
1: -14.2732544, -11.7366562, -14.2732496, -11.7366638, -1.8241062, 1.8241029
2: -8.4603930, -6.5364137, -8.4603920, -6.5364165, -1.3472133, 1.3472147
3: -10.8424568, -8.4108410, -10.8424587, -8.4108448, -2.1505728, 2.1505666
4: -4.4520521, -2.5586674, -4.4520473, -2.5586717, -1.6091599, 1.6091549
5: -2.9136353, -1.0502625, -2.9136362, -1.0502634, -1.4807296, 1.4807291
6: -10.6823807, -7.8721638, -10.6823845, -7.8721642, -1.7176204, 1.7176228
7: -1.8836951, 0.1442804, -1.8836939, 0.1442804, -1.8347454, 1.8347435
8: -4.4789991, -2.3308210, -4.4790015, -2.3308239, -1.4949732, 1.4949772
9: 1.3788483, 2.6902061, 1.3788505, 2.6902032, -1.0200336, 1.0200299

Time for backsubstitution: 22.62 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1942
type: A, layer: 3, pos: 766
type: A, layer: 3, pos: 2228
type: A, layer: 3, pos: 2382
type: A, layer: 3, pos: 1683
type: A, layer: 3, pos: 2627
type: A, layer: 3, pos: 1682
type: A, layer: 3, pos: 556
type: A, layer: 3, pos: 1109
type: A, layer: 3, pos: 2460
type: A, layer: 3, pos: 765
type: A, layer: 3, pos: 1684
type: A, layer: 3, pos: 1452
type: A, layer: 3, pos: 2544
type: A, layer: 3, pos: 2525
type: A, layer: 3, pos: 2825
type: A, layer: 3, pos: 1101
type: A, layer: 3, pos: 758
type: A, layer: 3, pos: 3104
type: A, layer: 3, pos: 2152
type: A, layer: 3, pos: 1143
type: A, layer: 3, pos: 1920
type: A, layer: 3, pos: 1731
type: A, layer: 3, pos: 1383
type: A, layer: 3, pos: 1151
type: A, layer: 3, pos: 1494
type: A, layer: 3, pos: 422
type: A, layer: 3, pos: 2518
type: A, layer: 3, pos: 46
type: A, layer: 3, pos: 1788
type: A, layer: 3, pos: 2909
type: A, layer: 3, pos: 1692
type: A, layer: 3, pos: 1843
type: A, layer: 3, pos: 920
type: A, layer: 3, pos: 311
type: A, layer: 3, pos: 921
type: A, layer: 3, pos: 563
type: A, layer: 3, pos: 913
type: A, layer: 3, pos: 1503
type: A, layer: 3, pos: 1985
type: A, layer: 3, pos: 2607
type: A, layer: 3, pos: 907
type: A, layer: 3, pos: 2145
type: A, layer: 3, pos: 897
type: A, layer: 3, pos: 598
type: A, layer: 3, pos: 1104
type: A, layer: 3, pos: 173
type: A, layer: 3, pos: 158
type: A, layer: 3, pos: 1773
type: A, layer: 3, pos: 1492
type: A, layer: 3, pos: 904

Time for candidate selection: 0.56 seconds

### Candidate
type: A, layer: 3, pos: 1942

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5779445, upper bound: 0.5903620
time: 4.02 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 9, lower bound: -0.5863867, upper bound: 0.5903637
time: 3.82 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 59.55 + 544.00 = 603.55 seconds
