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
execution time: IAR + RelationalAnalysis = 23.48 + 34.33 = 57.81 seconds
status: Status.UNKNOWN
relational distance
Output dim: 8, lower bound: -1.0405798, upper bound: 1.0405789

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5805
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5805

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405785, upper bound: 1.0396753
time: 5.38 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405785, upper bound: 1.0405775
time: 4.83 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 10.35 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 10.35
Output dim: 8, lower bound: -1.0405785, upper bound: 1.0396753
NS_A2, status: Status.UNKNOWN, split count: 1, time: 10.35
Output dim: 8, lower bound: -1.0405785, upper bound: 1.0405775

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -6.1727223, -1.9947743, -6.1833200, -1.9944155, -3.2104397, 3.2205067
1: -12.2193813, -8.9918146, -12.2266054, -8.9915438, -2.8150358, 2.8219624
2: -5.6021357, -2.2392821, -5.6109948, -2.2374420, -2.7850418, 2.7923903
3: -5.3690166, -1.5078948, -5.3699169, -1.5061395, -3.5507689, 3.5499167
4: -11.5189695, -7.5783062, -11.5212889, -7.5732951, -3.3237209, 3.3208966
5: -6.2878013, -3.0851054, -6.2888985, -3.0835276, -2.6664743, 2.6654387
6: -12.4260101, -8.6976948, -12.4269085, -8.6967430, -2.9239368, 2.9233985
7: -8.1650391, -4.6791401, -8.1675873, -4.6758566, -3.4891825, 3.4884472
8: 7.7414160, 10.0551033, 7.7401800, 10.0575123, -2.2146711, 2.2132986
9: -6.3445268, -2.8050721, -6.3461952, -2.8040040, -2.9266605, 2.9272366

Time for backsubstitution: 21.78 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400844, upper bound: 1.0382997
time: 6.10 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405771, upper bound: 1.0396735
time: 5.16 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -6.2037172, -1.9477444, -6.1948833, -1.9940338, -3.2338514, 3.2736328
1: -12.2469521, -8.9744186, -12.2345181, -8.9912519, -2.8427224, 2.8481331
2: -5.6229401, -2.1984048, -5.6206656, -2.2354527, -2.7997551, 2.8347559
3: -5.3863258, -1.5006931, -5.3708882, -1.5042098, -3.5699253, 3.5663614
4: -11.5466633, -7.5620065, -11.5237999, -7.5678220, -3.3563213, 3.3381400
5: -6.3025026, -3.0794792, -6.2900906, -3.0817852, -2.6854024, 2.6848931
6: -12.4390488, -8.6935806, -12.4278793, -8.6957169, -2.9402156, 2.9468064
7: -8.1877203, -4.6719470, -8.1703529, -4.6722755, -3.5154448, 3.4984059
8: 7.7302880, 10.0681181, 7.7388430, 10.0601482, -2.2405238, 2.2305124
9: -6.3509598, -2.7878101, -6.3480201, -2.8028452, -2.9352970, 2.9539528

Time for backsubstitution: 21.83 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 540
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 540

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400844, upper bound: 1.0391545
time: 4.88 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405771, upper bound: 1.0405757
time: 4.68 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 31.52 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 31.52
Output dim: 8, lower bound: -1.0400844, upper bound: 1.0382997
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 31.52
Output dim: 8, lower bound: -1.0405771, upper bound: 1.0396735
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 31.52
Output dim: 8, lower bound: -1.0400844, upper bound: 1.0391545
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 31.52
Output dim: 8, lower bound: -1.0405771, upper bound: 1.0405757

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -6.1710367, -2.0061798, -6.1727204, -2.0183611, -3.1844077, 3.1957388
1: -12.2138634, -8.9943295, -12.2128792, -8.9978924, -2.8022714, 2.8055267
2: -5.6008654, -2.2489288, -5.6049070, -2.2587013, -2.7617922, 2.7751722
3: -5.3610716, -1.5101209, -5.3515339, -1.5153465, -3.5328541, 3.5284414
4: -11.5127773, -7.5804257, -11.5045643, -7.5820684, -3.3070583, 3.3013711
5: -6.2863026, -3.0952916, -6.2802458, -3.1049480, -2.6420927, 2.6449251
6: -12.4237900, -8.7036343, -12.4191055, -8.7096901, -2.9083776, 2.9094825
7: -8.1574049, -4.6802788, -8.1481075, -4.6792092, -3.4781957, 3.4678288
8: 7.7466345, 10.0527401, 7.7519464, 10.0501795, -2.1989160, 2.1984005
9: -6.3417892, -2.8101735, -6.3345156, -2.8166437, -2.9104242, 2.9104376

Time for backsubstitution: 23.17 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of NS_A1_B1_B1

### Relational analysis result of NS_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392524, upper bound: 1.0383029
time: 6.13 seconds

## Relational analysis of NS_A1_B1_B2

### Relational analysis result of NS_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392524, upper bound: 1.0383029
time: 5.75 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -6.1727204, -1.9947753, -6.1833167, -1.9944234, -3.1927977, 3.2205043
1: -12.2193813, -8.9918165, -12.2266016, -8.9915457, -2.8182449, 2.8217893
2: -5.6021366, -2.2392821, -5.6109939, -2.2374480, -2.7824345, 2.7923884
3: -5.3690162, -1.5078964, -5.3699126, -1.5061409, -3.5507650, 3.5420942
4: -11.5189676, -7.5783086, -11.5212860, -7.5732946, -3.3237171, 3.3185577
5: -6.2878008, -3.0851059, -6.2888975, -3.0835323, -2.6512232, 2.6654377
6: -12.4260101, -8.6976967, -12.4269085, -8.6967487, -2.9239163, 2.9237514
7: -8.1650381, -4.6791401, -8.1675844, -4.6758556, -3.4891825, 3.4884443
8: 7.7414155, 10.0551023, 7.7401819, 10.0575104, -2.2178450, 2.2131269
9: -6.3445277, -2.8050723, -6.3461938, -2.8040071, -2.9226360, 2.9272366

Time for backsubstitution: 23.24 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 5805

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396842, upper bound: 1.0396742
time: 5.55 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396842, upper bound: 1.0396755
time: 5.79 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -6.2020307, -1.9591286, -6.1842809, -2.0179725, -3.2078180, 3.2449837
1: -12.2414618, -8.9769354, -12.2208109, -8.9975929, -2.8300400, 2.8316727
2: -5.6216559, -2.2080779, -5.6145763, -2.2567010, -2.7764835, 2.8138125
3: -5.3784513, -1.5029225, -5.3525047, -1.5134199, -3.5520716, 3.5449152
4: -11.5405006, -7.5641446, -11.5071220, -7.5765982, -3.3397293, 3.3186955
5: -6.3009987, -3.0896363, -6.2814341, -3.1031876, -2.6610260, 2.6644073
6: -12.4368248, -8.6995020, -12.4200706, -8.7086496, -2.9245996, 2.9329166
7: -8.1801300, -4.6730857, -8.1508923, -4.6756277, -3.5045023, 3.4778066
8: 7.7354960, 10.0657606, 7.7506075, 10.0528059, -2.2247953, 2.2156000
9: -6.3482027, -2.7927990, -6.3363285, -2.8154972, -2.9190378, 2.9372568

Time for backsubstitution: 23.14 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.11 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of NS_A2_B1_B1

### Relational analysis result of NS_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0366394, upper bound: 1.0388472
time: 6.12 seconds

## Relational analysis of NS_A2_B1_B2

### Relational analysis result of NS_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400772, upper bound: 1.0391475
time: 5.02 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -6.2037177, -1.9477453, -6.1948795, -1.9940362, -3.2162299, 3.2686148
1: -12.2469482, -8.9744177, -12.2345142, -8.9912519, -2.8459330, 2.8479609
2: -5.6229410, -2.1984069, -5.6206646, -2.2354541, -2.7971478, 2.8320181
3: -5.3863239, -1.5006931, -5.3708844, -1.5042095, -3.5699244, 3.5585403
4: -11.5466633, -7.5620065, -11.5237970, -7.5678225, -3.3563185, 3.3358011
5: -6.3025017, -3.0794806, -6.2900915, -3.0817900, -2.6701522, 2.6848907
6: -12.4390478, -8.6935806, -12.4278793, -8.6957169, -2.9401927, 2.9471588
7: -8.1877193, -4.6719475, -8.1703491, -4.6722755, -3.5154438, 3.4984016
8: 7.7302895, 10.0681171, 7.7388449, 10.0601473, -2.2436996, 2.2303429
9: -6.3509588, -2.7878122, -6.3480210, -2.8028481, -2.9312739, 2.9539518

Time for backsubstitution: 23.19 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540

Time for candidate selection: 0.13 seconds

### Candidate
type: B, layer: 1, pos: 4598

## Relational analysis of NS_A2_B2_B1

### Relational analysis result of NS_A2_B2_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0371257, upper bound: 1.0402592
time: 8.34 seconds

## Relational analysis of NS_A2_B2_B2

### Relational analysis result of NS_A2_B2_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405698, upper bound: 1.0405687
time: 4.70 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 36.37 seconds
NS_A1_B1_B1, status: Status.UNKNOWN, split count: 3, time: 36.37
Output dim: 8, lower bound: -1.0392524, upper bound: 1.0383029
NS_A1_B1_B2, status: Status.UNKNOWN, split count: 3, time: 36.37
Output dim: 8, lower bound: -1.0392524, upper bound: 1.0383029
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 36.37
Output dim: 8, lower bound: -1.0396842, upper bound: 1.0396742
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 36.37
Output dim: 8, lower bound: -1.0396842, upper bound: 1.0396755
NS_A2_B1_B1, status: Status.UNKNOWN, split count: 3, time: 36.37
Output dim: 8, lower bound: -1.0366394, upper bound: 1.0388472
NS_A2_B1_B2, status: Status.UNKNOWN, split count: 3, time: 36.37
Output dim: 8, lower bound: -1.0400772, upper bound: 1.0391475
NS_A2_B2_B1, status: Status.UNKNOWN, split count: 3, time: 36.37
Output dim: 8, lower bound: -1.0371257, upper bound: 1.0402592
NS_A2_B2_B2, status: Status.UNKNOWN, split count: 3, time: 36.37
Output dim: 8, lower bound: -1.0405698, upper bound: 1.0405687

## BFS NS instance: NS_A1_B1_B1

### Backsubstitution after applying NS history:
0: -6.1710367, -2.0061798, -6.1621289, -2.0187168, -3.1839519, 3.1852098
1: -12.2138634, -8.9943295, -12.2056417, -8.9981728, -2.8019667, 2.7982950
2: -5.6008654, -2.2489288, -5.5960598, -2.2605486, -2.7601480, 2.7661929
3: -5.3610716, -1.5101209, -5.3506393, -1.5171008, -3.5293331, 3.5257516
4: -11.5127773, -7.5804257, -11.5021973, -7.5870790, -3.3020697, 3.2991486
5: -6.2863026, -3.0952916, -6.2791481, -3.1065397, -2.6380968, 2.6419835
6: -12.4237900, -8.7036343, -12.4182110, -8.7106514, -2.9042344, 2.9059181
7: -8.1574049, -4.6802788, -8.1455431, -4.6824923, -3.4749126, 3.4652643
8: 7.7466345, 10.0527401, 7.7531824, 10.0477810, -2.1946301, 2.1954768
9: -6.3417892, -2.8101735, -6.3328571, -2.8177018, -2.9080877, 2.9075203

Time for backsubstitution: 22.53 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of NS_A1_B1_B1_A1

### Relational analysis result of NS_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0389545, upper bound: 1.0348678
time: 6.41 seconds

## Relational analysis of NS_A1_B1_B1_A2

### Relational analysis result of NS_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392454, upper bound: 1.0382933
time: 6.46 seconds

## BFS NS instance: NS_A1_B1_B2

### Backsubstitution after applying NS history:
0: -6.1710367, -2.0061798, -6.1931109, -1.9716363, -3.2256050, 3.2155094
1: -12.2138634, -8.9943295, -12.2332678, -8.9807749, -2.8201804, 2.8267331
2: -5.6008654, -2.2489288, -5.6168251, -2.2197075, -2.7926831, 2.7878542
3: -5.3610716, -1.5101209, -5.3681040, -1.5099034, -3.5377016, 3.5412393
4: -11.5127773, -7.5804257, -11.5300026, -7.5708046, -3.3186865, 3.3264494
5: -6.2863026, -3.0952916, -6.2938480, -3.1008792, -2.6442595, 2.6565514
6: -12.4237900, -8.7036343, -12.4312429, -8.7065039, -2.9099851, 2.9176049
7: -8.1574049, -4.6802788, -8.1683245, -4.6752987, -3.4821062, 3.4880457
8: 7.7466345, 10.0527401, 7.7420259, 10.0607948, -2.2086110, 2.2072260
9: -6.3417892, -2.8101735, -6.3392596, -2.8001912, -2.9254861, 2.9135695

Time for backsubstitution: 21.96 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of NS_A1_B1_B2_A1

### Relational analysis result of NS_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0389545, upper bound: 1.0348653
time: 6.81 seconds

## Relational analysis of NS_A1_B1_B2_A2

### Relational analysis result of NS_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392454, upper bound: 1.0382956
time: 5.75 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -6.1727204, -1.9947753, -6.1727219, -1.9947777, -3.1923122, 3.2099710
1: -12.2193813, -8.9918165, -12.2193813, -8.9918175, -2.8179264, 2.8145442
2: -5.6021366, -2.2392821, -5.6021366, -2.2392840, -2.7807961, 2.7834020
3: -5.3690162, -1.5078964, -5.3690133, -1.5078962, -3.5472412, 3.5394201
4: -11.5189676, -7.5783086, -11.5189676, -7.5783081, -3.3187218, 3.3163857
5: -6.2878008, -3.0851059, -6.2878036, -3.0851102, -2.6472502, 2.6624994
6: -12.4260101, -8.6976967, -12.4260111, -8.6976995, -2.9197984, 2.9201732
7: -8.1650381, -4.6791401, -8.1650372, -4.6791420, -3.4858961, 3.4858971
8: 7.7414155, 10.0551023, 7.7414169, 10.0551033, -2.2135510, 2.2102041
9: -6.3445277, -2.8050723, -6.3445268, -2.8050749, -2.9202847, 2.9243069

Time for backsubstitution: 23.03 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0393653, upper bound: 1.0362208
time: 6.17 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396772, upper bound: 1.0396670
time: 5.54 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -6.1727204, -1.9947753, -6.2037158, -1.9477487, -3.2330875, 3.2402868
1: -12.2193813, -8.9918165, -12.2469473, -8.9744177, -2.8361592, 2.8429556
2: -5.6021366, -2.2392821, -5.6229391, -2.1984091, -2.8126397, 2.8051295
3: -5.3690162, -1.5078964, -5.3863220, -1.5006948, -3.5556126, 3.5547457
4: -11.5189676, -7.5783086, -11.5466604, -7.5620117, -3.3353262, 3.3435335
5: -6.2878008, -3.0851059, -6.3025017, -3.0794830, -2.6533566, 2.6771097
6: -12.4260101, -8.6976967, -12.4390478, -8.6935844, -2.9254646, 2.9319196
7: -8.1650381, -4.6791401, -8.1877155, -4.6719465, -3.4930916, 3.5085754
8: 7.7414155, 10.0551023, 7.7302904, 10.0681171, -2.2275877, 2.2218883
9: -6.3445277, -2.8050723, -6.3509574, -2.7878125, -2.9374142, 2.9303856

Time for backsubstitution: 22.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 4598
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 4598
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540

Time for candidate selection: 0.10 seconds

### Candidate
type: A, layer: 1, pos: 4598

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0393653, upper bound: 1.0362215
time: 5.98 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0396772, upper bound: 1.0396674
time: 5.30 seconds

## BFS NS instance: NS_A2_B1_B1

### Backsubstitution after applying NS history:
0: -6.1981802, -1.9683328, -6.1607890, -2.0684378, -3.1539850, 3.1821465
1: -12.2353134, -8.9773941, -12.1877918, -9.0012579, -2.8122835, 2.7997561
2: -5.6113300, -2.2098477, -5.5576658, -2.2692349, -2.7398925, 2.7551806
3: -5.3678675, -1.5050740, -5.2942629, -1.5267141, -3.5259581, 3.4847951
4: -11.5385675, -7.5877733, -11.4917774, -7.7064552, -3.2082214, 3.2323160
5: -6.2992902, -3.0979605, -6.2698288, -3.1490221, -2.6128430, 2.6326070
6: -12.4336767, -8.7003899, -12.4040499, -8.7145443, -2.9144783, 2.9176216
7: -8.1776352, -4.6757998, -8.1355877, -4.6876945, -3.4899406, 3.4597878
8: 7.7374597, 10.0613365, 7.7618299, 10.0283518, -2.2011209, 2.1965616
9: -6.3464556, -2.8093951, -6.3232336, -2.9069140, -2.8262577, 2.8799410

Time for backsubstitution: 22.88 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of NS_A2_B1_B1_A1

### Relational analysis result of NS_A2_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0353388, upper bound: 1.0388428
time: 5.32 seconds

## Relational analysis of NS_A2_B1_B1_A2

### Relational analysis result of NS_A2_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0366375, upper bound: 1.0388431
time: 12.01 seconds

## BFS NS instance: NS_A2_B1_B2

### Backsubstitution after applying NS history:
0: -6.2020307, -1.9591286, -6.1842790, -2.0179734, -3.1840534, 3.2431502
1: -12.2414618, -8.9769354, -12.2208090, -8.9975920, -2.8290062, 2.8499913
2: -5.6216559, -2.2080779, -5.6145716, -2.2567019, -2.7764826, 2.7670627
3: -5.3784513, -1.5029225, -5.3525014, -1.5134208, -3.5520716, 3.5244894
4: -11.5405006, -7.5641446, -11.5071220, -7.5766044, -3.2317324, 3.3186951
5: -6.3009987, -3.0896363, -6.2814341, -3.1031914, -2.6441793, 2.6644058
6: -12.4368248, -8.6995020, -12.4200735, -8.7086525, -2.9246011, 2.9279957
7: -8.1801300, -4.6730857, -8.1508942, -4.6756277, -3.5045023, 3.4778085
8: 7.7354960, 10.0657606, 7.7506075, 10.0528049, -2.2244678, 2.2153542
9: -6.3482027, -2.7927990, -6.3363290, -2.8155019, -2.8603330, 2.9372578

Time for backsubstitution: 22.98 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.11 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of NS_A2_B1_B2_A1

### Relational analysis result of NS_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0387241, upper bound: 1.0391457
time: 4.91 seconds

## Relational analysis of NS_A2_B1_B2_A2

### Relational analysis result of NS_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0400752, upper bound: 1.0391459
time: 4.78 seconds

## BFS NS instance: NS_A2_B2_B1

### Backsubstitution after applying NS history:
0: -6.1998587, -1.9569550, -6.1713524, -2.0445619, -3.1623774, 3.2056761
1: -12.2408047, -8.9748802, -12.2015257, -8.9949579, -2.8281531, 2.8160429
2: -5.6126127, -2.2001810, -5.5637479, -2.2480230, -2.7598968, 2.7733846
3: -5.3757420, -1.5028491, -5.3126192, -1.5175383, -3.5441341, 3.4983778
4: -11.5447302, -7.5856361, -11.5084829, -7.6976871, -3.2248154, 3.2488279
5: -6.3007951, -3.0878010, -6.2785149, -3.1275911, -2.6220083, 2.6542759
6: -12.4359026, -8.6944685, -12.4118347, -8.7015877, -2.9300432, 2.9318461
7: -8.1852140, -4.6746635, -8.1549635, -4.6843429, -3.5008712, 3.4802999
8: 7.7322531, 10.0636911, 7.7501011, 10.0356693, -2.2199960, 2.2112813
9: -6.3491945, -2.8044050, -6.3348703, -2.8942535, -2.8384991, 2.8972325

Time for backsubstitution: 23.61 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5805
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 4598
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540

Time for candidate selection: 0.12 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of NS_A2_B2_B1_B1

### Relational analysis result of NS_A2_B2_B1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0371239, upper bound: 1.0389517
time: 9.18 seconds

## Relational analysis of NS_A2_B2_B1_B2

### Relational analysis result of NS_A2_B2_B1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0371239, upper bound: 1.0402574
time: 8.62 seconds

## BFS NS instance: NS_A2_B2_B2

### Backsubstitution after applying NS history:
0: -6.2037177, -1.9477453, -6.1948810, -1.9940381, -3.1924686, 3.2667820
1: -12.2469482, -8.9744177, -12.2345133, -8.9912529, -2.8449020, 2.8662529
2: -5.6229410, -2.1984069, -5.6206608, -2.2354562, -2.7971487, 2.7853012
3: -5.3863239, -1.5006931, -5.3708835, -1.5042105, -3.5699234, 3.5381145
4: -11.5466633, -7.5620065, -11.5237989, -7.5678291, -3.2483234, 3.3358006
5: -6.3025017, -3.0794806, -6.2900925, -3.0817919, -2.6533051, 2.6848907
6: -12.4390478, -8.6935806, -12.4278793, -8.6957188, -2.9401913, 2.9422121
7: -8.1877193, -4.6719475, -8.1703491, -4.6722755, -3.5154438, 3.4984016
8: 7.7302895, 10.0681171, 7.7388444, 10.0601454, -2.2434144, 2.2300956
9: -6.3509588, -2.7878122, -6.3480206, -2.8028529, -2.8725700, 2.9539509

Time for backsubstitution: 23.69 seconds

### NS candidates at layer 1
type: A, layer: 1, pos: 5749
type: B, layer: 1, pos: 5749
type: B, layer: 1, pos: 891
type: B, layer: 1, pos: 6182
type: B, layer: 1, pos: 5805
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 835
type: B, layer: 1, pos: 133
type: A, layer: 1, pos: 891
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 133
type: A, layer: 1, pos: 6182
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 4598

Time for candidate selection: 0.12 seconds

### Candidate
type: A, layer: 1, pos: 5749

## Relational analysis of NS_A2_B2_B2_A1

### Relational analysis result of NS_A2_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0392314, upper bound: 1.0405690
time: 5.96 seconds

## Relational analysis of NS_A2_B2_B2_A2

### Relational analysis result of NS_A2_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0405679, upper bound: 1.0405668
time: 4.77 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 34.55 seconds
NS_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0389545, upper bound: 1.0348678
NS_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0392454, upper bound: 1.0382933
NS_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0389545, upper bound: 1.0348653
NS_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0392454, upper bound: 1.0382956
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0393653, upper bound: 1.0362208
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0396772, upper bound: 1.0396670
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0393653, upper bound: 1.0362215
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0396772, upper bound: 1.0396674
NS_A2_B1_B1_A1, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0353388, upper bound: 1.0388428
NS_A2_B1_B1_A2, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0366375, upper bound: 1.0388431
NS_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0387241, upper bound: 1.0391457
NS_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0400752, upper bound: 1.0391459
NS_A2_B2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0371239, upper bound: 1.0389517
NS_A2_B2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0371239, upper bound: 1.0402574
NS_A2_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0392314, upper bound: 1.0405690
NS_A2_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 34.55
Output dim: 8, lower bound: -1.0405679, upper bound: 1.0405668

## BFS NS instance: NS_A1_B1_B1_A1

### Backsubstitution after applying NS history:
0: -6.1474972, -2.0566831, -6.1582541, -2.0279164, -3.1522722, 3.1313086
1: -12.1807375, -8.9980097, -12.1994953, -8.9986296, -2.7699232, 2.7804890
2: -5.5439463, -2.2613964, -5.5857363, -2.2622881, -2.7015209, 2.7259221
3: -5.3028507, -1.5234334, -5.3400993, -1.5192549, -3.4691648, 3.5040240
4: -11.4974995, -7.7102976, -11.5002689, -7.6107073, -3.2143888, 3.1676683
5: -6.2747707, -3.1411271, -6.2774582, -3.1148696, -2.6169119, 2.5938535
6: -12.4078264, -8.7094917, -12.4151230, -8.7115345, -2.8889265, 2.8958817
7: -8.1421547, -4.6923418, -8.1430693, -4.6852069, -3.4569478, 3.4507275
8: 7.7578468, 10.0282173, 7.7551394, 10.0433578, -2.1755772, 2.1717343
9: -6.3286519, -2.9015939, -6.3311009, -2.8342903, -2.8628459, 2.8147340

Time for backsubstitution: 23.64 seconds

### NS candidates at layer 1
type: B, layer: 1, pos: 5749
type: A, layer: 1, pos: 5749
type: A, layer: 1, pos: 540
type: A, layer: 1, pos: 891
type: B, layer: 1, pos: 891
type: A, layer: 1, pos: 6182
type: B, layer: 1, pos: 835
type: A, layer: 1, pos: 835
type: B, layer: 1, pos: 6182
type: A, layer: 1, pos: 133
type: B, layer: 1, pos: 133
type: B, layer: 1, pos: 135
type: A, layer: 1, pos: 135
type: B, layer: 1, pos: 4598

Time for candidate selection: 0.14 seconds

### Candidate
type: B, layer: 1, pos: 5749

## Relational analysis of NS_A1_B1_B1_A1_B1

### Relational analysis result of NS_A1_B1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0389526, upper bound: 1.0335734
time: 8.36 seconds

## Relational analysis of NS_A1_B1_B1_A1_B2

### Relational analysis result of NS_A1_B1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 8, lower bound: -1.0389526, upper bound: 1.0348725
time: 6.37 seconds

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.81 + 542.71 = 600.52 seconds
