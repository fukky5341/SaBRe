## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.046875
Delta epsilon: 0.01171875
execution index: (3, 4, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 1.0301740019999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2329988, 3.2329993)
1: (-12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8305802, 2.8305807)
2: (-5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8056412, 2.8056412)
3: (-5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5602217, 3.5602221)
4: (-11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3337431, 3.3337431)
5: (-6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6769562, 2.6769567)
6: (-12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9359779, 2.9359775)
7: (-8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884)
8: (7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2254882, 2.2254882)
9: (-6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9353604, 2.9353604)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.95 + 34.52 = 57.48 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.0405798, upper bound: 1.0405789

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 4598

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405728, upper bound: 1.0375872
time: 5.88 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375846, upper bound: 1.0405753
time: 9.80 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 15.79 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 15.79
Output dim: 8, lower bound: -1.0405728, upper bound: 1.0375872
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 15.79
Output dim: 8, lower bound: -1.0375846, upper bound: 1.0405753

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2092361, 3.2058396
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8420048, 2.8488746
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7517409, 2.7592268
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5368919, 3.5398016
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2257471, 3.2103138
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6601110, 2.6577020
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9291964, 2.9310312
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2251902, 2.2215469
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8766561, 2.8682899

Time for backsubstitution: 22.17 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5805

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405715, upper bound: 1.0366685
time: 7.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396693, upper bound: 1.0375836
time: 7.66 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2058392, 3.2092361
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8488750, 2.8420043
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7592263, 2.7517409
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5398016, 3.5368924
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2103138, 3.2257471
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6577020, 2.6601114
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9310312, 2.9291959
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2215471, 2.2251899
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8682895, 2.8766561

Time for backsubstitution: 21.79 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 5805

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375833, upper bound: 1.0396696
time: 5.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0366685, upper bound: 1.0405716
time: 8.68 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 36.38 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 36.38
Output dim: 8, lower bound: -1.0405715, upper bound: 1.0366685
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 36.38
Output dim: 8, lower bound: -1.0396693, upper bound: 1.0375836
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 36.38
Output dim: 8, lower bound: -1.0375833, upper bound: 1.0396696
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 36.38
Output dim: 8, lower bound: -1.0366685, upper bound: 1.0405716

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1972828, 3.1984205
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8402786, 2.8478022
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7375407, 2.7504196
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5340595, 3.5352411
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2239981, 3.2074914
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6562510, 2.6514826
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9255247, 2.9251118
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2212811, 2.2191238
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8750215, 2.8656530

Time for backsubstitution: 22.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 540

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0397021, upper bound: 1.0366696
time: 7.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405701, upper bound: 1.0358042
time: 7.44 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2018166, 3.1938868
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8409328, 2.8471489
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7429338, 2.7450266
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5323324, 3.5369687
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2229242, 3.2085648
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6538916, 2.6538415
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9232769, 2.9273596
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2227664, 2.2176380
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8740191, 2.8666553

Time for backsubstitution: 23.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 540

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0388004, upper bound: 1.0375844
time: 5.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396679, upper bound: 1.0367197
time: 6.18 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1938868, 3.2018166
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8471489, 2.8409324
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7450271, 2.7429342
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5369682, 3.5323319
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2085648, 3.2229242
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6538410, 2.6538920
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9273596, 2.9232769
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2176380, 2.2227666
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8666558, 2.8740196

Time for backsubstitution: 22.97 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 540

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0367189, upper bound: 1.0396676
time: 9.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0375819, upper bound: 1.0388005
time: 9.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1984205, 3.1972833
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8478022, 2.8402786
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7504201, 2.7375407
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5352411, 3.5340595
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2074909, 3.2239981
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6514826, 2.6562510
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9251118, 2.9255247
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2191238, 2.2212811
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8656535, 2.8750219

Time for backsubstitution: 23.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 540

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0358042, upper bound: 1.0405691
time: 4.82 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0366671, upper bound: 1.0397013
time: 4.88 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 33.24 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.24
Output dim: 8, lower bound: -1.0397021, upper bound: 1.0366696
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.24
Output dim: 8, lower bound: -1.0405701, upper bound: 1.0358042
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.24
Output dim: 8, lower bound: -1.0388004, upper bound: 1.0375844
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.24
Output dim: 8, lower bound: -1.0396679, upper bound: 1.0367197
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.24
Output dim: 8, lower bound: -1.0367189, upper bound: 1.0396676
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.24
Output dim: 8, lower bound: -1.0375819, upper bound: 1.0388005
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 33.24
Output dim: 8, lower bound: -1.0358042, upper bound: 1.0405691
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 33.24
Output dim: 8, lower bound: -1.0366671, upper bound: 1.0397013

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1796637, 3.1776690
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8434887, 2.8505878
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7349348, 2.7474427
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5251226, 3.5274210
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2213287, 3.2051544
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6410027, 2.6340551
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9258318, 2.9254665
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2244582, 2.2218795
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8710003, 2.8610559

Time for backsubstitution: 22.98 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391772, upper bound: 1.0361495
time: 6.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382799, upper bound: 1.0361506
time: 5.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1765318, 3.1808009
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8430643, 2.8510127
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7345629, 2.7478147
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5262394, 3.5263047
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2216616, 3.2048221
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6388226, 2.6362338
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9258795, 2.9254198
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2240372, 2.2223003
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8704243, 2.8616314

Time for backsubstitution: 22.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400170, upper bound: 1.0352893
time: 5.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391479, upper bound: 1.0352877
time: 5.75 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1841974, 3.1731358
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8441429, 2.8499341
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7403288, 2.7420492
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5233955, 3.5291486
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2202549, 3.2062283
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6386433, 2.6364136
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9235849, 2.9277143
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2259436, 2.2203939
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8699980, 2.8620586

Time for backsubstitution: 23.16 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.09 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382749, upper bound: 1.0361535
time: 8.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382764, upper bound: 1.0370377
time: 5.80 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1810656, 3.1762671
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8437176, 2.8503594
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7399569, 2.7424216
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5245123, 3.5280323
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2205877, 3.2058954
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6364641, 2.6385927
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9236317, 2.9276671
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2255225, 2.2208147
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8694220, 2.8626342

Time for backsubstitution: 22.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.12 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391429, upper bound: 1.0352904
time: 9.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391444, upper bound: 1.0361898
time: 6.76 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1762667, 3.1810660
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8503590, 2.8437176
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7424212, 2.7399569
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5280323, 3.5245118
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2058954, 3.2205877
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6385927, 2.6364641
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9276667, 2.9236317
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2208152, 2.2255225
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8626337, 2.8694220

Time for backsubstitution: 23.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361894, upper bound: 1.0391444
time: 9.07 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0352903, upper bound: 1.0391428
time: 7.83 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1731358, 3.1841969
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8499336, 2.8441424
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7420492, 2.7403288
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5291491, 3.5233955
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2062283, 3.2202549
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6364136, 2.6386433
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9277143, 2.9235845
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2203941, 2.2259433
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8620586, 2.8699980

Time for backsubstitution: 23.73 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.11 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0370374, upper bound: 1.0382788
time: 5.63 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361533, upper bound: 1.0382746
time: 7.14 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1808004, 3.1765323
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8510132, 2.8430638
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7478151, 2.7345634
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5263052, 3.5262394
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2048225, 3.2216616
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6362343, 2.6388230
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9254198, 2.9258790
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2223005, 2.2240369
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8616314, 2.8704243

Time for backsubstitution: 23.77 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.10 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0352853, upper bound: 1.0391503
time: 6.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0352868, upper bound: 1.0400194
time: 5.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1776695, 3.1796632
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8505878, 2.8434892
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7474422, 2.7349353
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5274210, 3.5251231
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2051544, 3.2213287
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6340551, 2.6410022
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9254665, 2.9258318
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2218800, 2.2244577
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8610563, 2.8710003

Time for backsubstitution: 22.75 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.08 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361482, upper bound: 1.0382811
time: 5.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0361498, upper bound: 1.0391784
time: 5.39 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 34.06 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0391772, upper bound: 1.0361495
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0382799, upper bound: 1.0361506
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0400170, upper bound: 1.0352893
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0391479, upper bound: 1.0352877
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0382749, upper bound: 1.0361535
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0382764, upper bound: 1.0370377
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0391429, upper bound: 1.0352904
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0391444, upper bound: 1.0361898
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0361894, upper bound: 1.0391444
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0352903, upper bound: 1.0391428
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0370374, upper bound: 1.0382788
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0361533, upper bound: 1.0382746
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0352853, upper bound: 1.0391503
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0352868, upper bound: 1.0400194
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0361482, upper bound: 1.0382811
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 34.06
Output dim: 8, lower bound: -1.0361498, upper bound: 1.0391784

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.1786060, 3.1779346
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8434048, 2.8506083
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7335672, 2.7477903
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5251951, 3.5271444
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2214355, 3.2047348
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6411152, 2.6336012
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9260006, 2.9247971
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2241416, 2.2219605
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8710113, 2.8610206

Time for backsubstitution: 23.54 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.48 + 554.22 = 611.70 seconds
