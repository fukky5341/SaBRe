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
execution time: IAR + RelationalAnalysis = 22.95 + 34.94 = 57.89 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.0405798, upper bound: 1.0405789

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 835
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 540

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 835

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392584, upper bound: 1.0400687
time: 5.46 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392609, upper bound: 1.0392572
time: 4.84 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.32 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.32
Output dim: 8, lower bound: -1.0392584, upper bound: 1.0400687
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.32
Output dim: 8, lower bound: -1.0392609, upper bound: 1.0392572

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2327433, 3.2325673
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8270068, 2.8284712
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8056412, 2.8056836
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5585642, 3.5574160
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3328705, 3.3322587
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6741190, 2.6752825
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9341412, 2.9348912
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2254872, 2.2256858
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9345894, 2.9340534

Time for backsubstitution: 21.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 133

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0387767, upper bound: 1.0400688
time: 5.02 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392581, upper bound: 1.0395873
time: 4.39 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2325678, 3.2327433
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8284707, 2.8270063
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8056841, 2.8056412
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5574160, 3.5585642
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3322582, 3.3328695
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6752825, 2.6741190
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9348917, 2.9341407
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2256856, 2.2254870
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9340534, 2.9345889

Time for backsubstitution: 20.94 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 135

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0394705, upper bound: 1.0392581
time: 6.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400697, upper bound: 1.0386576
time: 4.80 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.75 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.75
Output dim: 8, lower bound: -1.0387767, upper bound: 1.0400688
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.75
Output dim: 8, lower bound: -1.0392581, upper bound: 1.0395873
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.75
Output dim: 8, lower bound: -1.0394705, upper bound: 1.0392581
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.75
Output dim: 8, lower bound: -1.0400697, upper bound: 1.0386576

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2328639, 3.2326732
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8267002, 2.8281193
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8056431, 2.8056269
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5598478, 3.5585508
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3344755, 3.3340955
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6741323, 2.6752968
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9334803, 2.9343038
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2252030, 2.2254367
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9359841, 2.9352598

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 135

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4598

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0387697, upper bound: 1.0370787
time: 7.99 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0357832, upper bound: 1.0400626
time: 6.89 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2328486, 3.2326884
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8266544, 2.8281651
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8055840, 2.8056850
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5596991, 3.5586996
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3347063, 3.3338656
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6741343, 2.6752949
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9335527, 2.9342303
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2252378, 2.2254019
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9357953, 2.9354486

Time for backsubstitution: 22.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 135

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0386585, upper bound: 1.0395907
time: 6.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392579, upper bound: 1.0389894
time: 5.09 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2325678, 3.2327428
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8284683, 2.8270040
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8056831, 2.8056402
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5574074, 3.5585561
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3322582, 3.3328681
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6752758, 2.6741114
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9348845, 2.9341331
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2256846, 2.2254863
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9340534, 2.9345889

Time for backsubstitution: 23.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 5749

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 891

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0388788, upper bound: 1.0382670
time: 5.83 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0384705, upper bound: 1.0387105
time: 4.65 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2325678, 3.2327433
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8284683, 2.8270040
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8056831, 2.8056402
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5574074, 3.5585546
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3322582, 3.3328681
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6752739, 2.6741123
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9348836, 2.9341335
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2256851, 2.2254860
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9340534, 2.9345889

Time for backsubstitution: 22.81 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5805

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400683, upper bound: 1.0377526
time: 10.47 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391660, upper bound: 1.0386561
time: 5.01 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 38.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 8, lower bound: -1.0387697, upper bound: 1.0370787
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 8, lower bound: -1.0357832, upper bound: 1.0400626
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 8, lower bound: -1.0386585, upper bound: 1.0395907
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 8, lower bound: -1.0392579, upper bound: 1.0389894
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 8, lower bound: -1.0388788, upper bound: 1.0382670
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 8, lower bound: -1.0384705, upper bound: 1.0387105
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 8, lower bound: -1.0400683, upper bound: 1.0377526
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 38.31
Output dim: 8, lower bound: -1.0391660, upper bound: 1.0386561

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2091036, 3.2055154
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8381238, 2.8464131
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7517424, 2.7592125
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5365162, 3.5381289
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2264814, 3.2106676
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6572876, 2.6560431
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9266987, 2.9293575
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2249041, 2.2214949
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8772802, 2.8681898

Time for backsubstitution: 23.13 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 540

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6182

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0380321, upper bound: 1.0370766
time: 11.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0387681, upper bound: 1.0363297
time: 5.94 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2057066, 3.2089119
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8449941, 2.8395429
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7592278, 2.7517271
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5394249, 3.5352192
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2110481, 3.2261009
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6548777, 2.6584525
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9285336, 2.9275222
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2212610, 2.2251377
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8689137, 2.8765559

Time for backsubstitution: 22.67 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5749

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0351677, upper bound: 1.0400631
time: 6.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0351687, upper bound: 1.0386454
time: 6.49 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2328486, 3.2326880
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8266525, 2.8281631
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8055840, 2.8056850
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5596905, 3.5586920
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3347073, 3.3338666
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6741285, 2.6752882
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9335480, 2.9342241
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2252364, 2.2254004
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9357963, 2.9354496

Time for backsubstitution: 22.88 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 540

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383854, upper bound: 1.0395854
time: 4.87 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0383929, upper bound: 1.0387180
time: 5.34 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2328486, 3.2326880
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8266525, 2.8281627
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8055840, 2.8056850
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5596914, 3.5586910
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3347073, 3.3338666
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6741266, 2.6752892
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9335470, 2.9342251
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2252364, 2.2254002
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9357963, 2.9354496

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 5805

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6182

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0385253, upper bound: 1.0389909
time: 7.53 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392564, upper bound: 1.0382345
time: 7.56 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2315121, 3.2330070
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8283844, 2.8270226
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8043146, 2.8059883
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5574732, 3.5582762
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3323669, 3.3324509
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6753860, 2.6736584
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9350481, 2.9334607
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2253680, 2.2255666
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9340634, 2.9345522

Time for backsubstitution: 21.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 540

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5749

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0374582, upper bound: 1.0376599
time: 4.91 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0388769, upper bound: 1.0376587
time: 4.45 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2325678, 3.2316871
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8284683, 2.8269196
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.8056831, 2.8042717
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5571280, 3.5585561
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3318405, 3.3328681
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6748223, 2.6741114
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9342117, 2.9341331
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2256846, 2.2251694
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9340167, 2.9345889

Time for backsubstitution: 21.86 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 6182

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4598

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0384635, upper bound: 1.0357277
time: 7.13 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0354803, upper bound: 1.0387052
time: 5.46 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2206125, 3.2253227
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8267422, 2.8259315
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7914829, 2.7968326
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5545759, 3.5539956
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3305092, 3.3300462
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6714144, 2.6678939
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9312134, 2.9282155
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2217755, 2.2230620
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9324198, 2.9319534

Time for backsubstitution: 21.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 891
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 4598

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 5749

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0386488, upper bound: 1.0371436
time: 6.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400665, upper bound: 1.0371396
time: 7.33 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2251463, 3.2207890
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8273964, 2.8252773
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7968750, 2.7914395
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5528498, 3.5557232
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.3294353, 3.3311200
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6690559, 2.6702523
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9289656, 2.9304633
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2232614, 2.2215765
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.9314175, 2.9329557

Time for backsubstitution: 21.39 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 133
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 4598
type: DSZ, layer: 1, pos: 6182
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 540

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0382972, upper bound: 1.0383911
time: 17.30 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0391646, upper bound: 1.0383859
time: 5.39 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 44.10 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0380321, upper bound: 1.0370766
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0387681, upper bound: 1.0363297
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0351677, upper bound: 1.0400631
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0351687, upper bound: 1.0386454
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0383854, upper bound: 1.0395854
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0383929, upper bound: 1.0387180
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0385253, upper bound: 1.0389909
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0392564, upper bound: 1.0382345
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0374582, upper bound: 1.0376599
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0388769, upper bound: 1.0376587
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0384635, upper bound: 1.0357277
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0354803, upper bound: 1.0387052
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0386488, upper bound: 1.0371436
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0400665, upper bound: 1.0371396
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0382972, upper bound: 1.0383911
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 44.10
Output dim: 8, lower bound: -1.0391646, upper bound: 1.0383859

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -6.1949110, -1.9940329, -6.1949110, -1.9940329, -3.2048178, 3.2037930
1: -12.2345352, -8.9912510, -12.2345352, -8.9912510, -2.8359189, 2.8455238
2: -5.6206837, -2.2354488, -5.6206837, -2.2354488, -2.7509308, 2.7571945
3: -5.3708911, -1.5042067, -5.3708911, -1.5042067, -3.5352249, 3.5376129
4: -11.5238056, -7.5678167, -11.5238056, -7.5678167, -3.2256460, 3.2085762
5: -6.2900953, -3.0817800, -6.2900953, -3.0817800, -2.6550417, 2.6551452
6: -12.4278812, -8.6957111, -12.4278812, -8.6957111, -2.9264545, 2.9292598
7: -8.1703577, -4.6722693, -8.1703577, -4.6722693, -3.4980884, 3.4980884
8: 7.7388391, 10.0601549, 7.7388391, 10.0601549, -2.2247481, 2.2211082
9: -6.3480263, -2.8028452, -6.3480263, -2.8028452, -2.8756227, 2.8640389

Time for backsubstitution: 21.71 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 540
type: DSZ, layer: 1, pos: 5805
type: DSZ, layer: 1, pos: 5749
type: DSZ, layer: 1, pos: 135
type: DSZ, layer: 1, pos: 891

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 540

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0371675, upper bound: 1.0370764
time: 5.73 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0380306, upper bound: 1.0362083
time: 9.00 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 36.45 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 36.45
Output dim: 8, lower bound: -1.0371675, upper bound: 1.0370764
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 36.45
Output dim: 8, lower bound: -1.0380306, upper bound: 1.0362083
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0387681, upper bound: 1.0363297
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0351677, upper bound: 1.0400631
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0351687, upper bound: 1.0386454
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0383854, upper bound: 1.0395854
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0383929, upper bound: 1.0387180
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0385253, upper bound: 1.0389909
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0392564, upper bound: 1.0382345
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0374582, upper bound: 1.0376599
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0388769, upper bound: 1.0376587
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0384635, upper bound: 1.0357277
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0354803, upper bound: 1.0387052
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0386488, upper bound: 1.0371436
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0400665, upper bound: 1.0371396
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0382972, upper bound: 1.0383911
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 36.45
Output dim: 8, lower bound: -1.0391646, upper bound: 1.0383859

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 57.89 + 543.56 = 601.45 seconds
