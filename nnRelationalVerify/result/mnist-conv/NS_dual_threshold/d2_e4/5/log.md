## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03125
Delta epsilon: 0.0078125
execution index: (2, 4, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.35854425


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8353262, 0.8353262)
1: (-17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7921734, 0.7921736)
2: (-6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8058171, 0.8058176)
3: (-13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5403173, 0.5403173)
4: (-5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0939751, 1.0939751)
5: (-6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5390172, 0.5390172)
6: (8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5590355, 0.5590355)
7: (-13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5208609, 0.5208609)
8: (-5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4545681, 0.4545680)
9: (-10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0445647, 1.0445647)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.75 + 35.09 = 57.84 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3983825, upper bound: 0.3983820

# Neuron Split (NS) starts

## BFS NS instance: NS

Time for backsubstitution: 0.00 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2130
type: B, layer: 3, pos: 2130
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.45 seconds

### Candidate
type: A, layer: 3, pos: 2130

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3828351, upper bound: 0.3830028
time: 3.51 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3830011, upper bound: 0.3830028
time: 3.43 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.41 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.41
Output dim: 6, lower bound: -0.3828351, upper bound: 0.3830028
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.41
Output dim: 6, lower bound: -0.3830011, upper bound: 0.3830028

## BFS NS instance: NS_A1

### Backsubstitution after applying NS history:
0: -1.3006644, -0.1919719, -1.3027682, -0.1919377, -0.8327065, 0.8352990
1: -17.6869717, -16.2289658, -17.6880779, -16.2289352, -0.7918634, 0.7921526
2: -6.2674885, -4.9662809, -6.2674923, -4.9595528, -0.8054919, 0.7981930
3: -13.7172203, -12.5216417, -13.7172546, -12.5187931, -0.5403080, 0.5374267
4: -5.2670107, -4.0625620, -5.2706857, -4.0625572, -1.0885086, 1.0937095
5: -6.8110862, -5.8979774, -6.8115540, -5.8979759, -0.5386009, 0.5388575
6: 8.7731228, 9.8564253, 8.7676487, 9.8564320, -0.5542858, 0.5590305
7: -13.7728157, -12.5742092, -13.7728157, -12.5704174, -0.5208571, 0.5174797
8: -5.8831196, -4.9708033, -5.8852777, -4.9708009, -0.4537847, 0.4545448
9: -10.3968105, -8.9332657, -10.3968801, -8.9319839, -1.0445342, 1.0440269

Time for backsubstitution: 8.35 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2130
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.23 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3802265, upper bound: 0.3802265
time: 6.23 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3802265, upper bound: 0.3830015
time: 4.71 seconds

## BFS NS instance: NS_A2

### Backsubstitution after applying NS history:
0: -1.2922771, -0.1785847, -1.2997128, -0.1920325, -0.8314743, 0.8504047
1: -17.6785374, -16.2226295, -17.6853809, -16.2290020, -0.7904811, 0.7920830
2: -6.3139429, -4.9936342, -6.2674856, -4.9699054, -0.8647919, 0.8039780
3: -13.7359467, -12.5246744, -13.7171612, -12.5204582, -0.5620506, 0.5398736
4: -5.2499752, -4.0363760, -5.2642794, -4.0625620, -1.0878248, 1.1235957
5: -6.8026385, -5.8952284, -6.8088913, -5.8979902, -0.5400434, 0.5384033
6: 8.8257484, 9.9217463, 8.7896423, 9.8564167, -0.5601008, 0.6441824
7: -13.7974424, -12.5833435, -13.7728148, -12.5741196, -0.5532346, 0.5206633
8: -5.8701897, -4.9533358, -5.8806214, -4.9708109, -0.4559369, 0.4564211
9: -10.4070578, -8.9496384, -10.3966942, -8.9374571, -1.0404482, 1.0411663

Time for backsubstitution: 8.39 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.25 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of NS_A2_A1

### Relational analysis result of NS_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3780658, upper bound: 0.3766561
time: 3.94 seconds

## Relational analysis of NS_A2_A2

### Relational analysis result of NS_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3786798, upper bound: 0.3786813
time: 3.38 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 15.97 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 15.97
Output dim: 6, lower bound: -0.3802265, upper bound: 0.3802265
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 15.97
Output dim: 6, lower bound: -0.3802265, upper bound: 0.3830015
NS_A2_A1, status: Status.UNKNOWN, split count: 2, time: 15.97
Output dim: 6, lower bound: -0.3780658, upper bound: 0.3766561
NS_A2_A2, status: Status.UNKNOWN, split count: 2, time: 15.97
Output dim: 6, lower bound: -0.3786798, upper bound: 0.3786813

## BFS NS instance: NS_A1_B1

### Backsubstitution after applying NS history:
0: -1.3006644, -0.1919719, -1.3006644, -0.1919719, -0.8326793, 0.8326793
1: -17.6869717, -16.2289658, -17.6869717, -16.2289658, -0.7918425, 0.7918425
2: -6.2674885, -4.9662809, -6.2674885, -4.9662809, -0.7978673, 0.7978673
3: -13.7172203, -12.5216417, -13.7172203, -12.5216417, -0.5374174, 0.5374174
4: -5.2670107, -4.0625620, -5.2670107, -4.0625620, -1.0882425, 1.0882425
5: -6.8110862, -5.8979774, -6.8110862, -5.8979774, -0.5384414, 0.5384413
6: 8.7731228, 9.8564253, 8.7731228, 9.8564253, -0.5542808, 0.5542805
7: -13.7728157, -12.5742092, -13.7728157, -12.5742092, -0.5174761, 0.5174761
8: -5.8831196, -4.9708033, -5.8831196, -4.9708033, -0.4537613, 0.4537613
9: -10.3968105, -8.9332657, -10.3968105, -8.9332657, -1.0439959, 1.0439954

Time for backsubstitution: 8.36 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.24 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3782938, upper bound: 0.3737713
time: 5.79 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3788568, upper bound: 0.3760782
time: 3.14 seconds

## BFS NS instance: NS_A1_B2

### Backsubstitution after applying NS history:
0: -1.3006644, -0.1919719, -1.2922771, -0.1785847, -0.8497963, 0.8285232
1: -17.6869717, -16.2289658, -17.6785374, -16.2226295, -0.7960622, 0.7796826
2: -6.2674885, -4.9662809, -6.3139429, -4.9936342, -0.7963719, 0.8620276
3: -13.7172203, -12.5216417, -13.7359467, -12.5246744, -0.5373929, 0.5604260
4: -5.2670107, -4.0625620, -5.2499752, -4.0363760, -1.1219873, 1.0814319
5: -6.8110862, -5.8979774, -6.8026385, -5.8952284, -0.5406418, 0.5309906
6: 8.7731228, 9.8564253, 8.8257484, 9.9217463, -0.6559501, 0.5345092
7: -13.7728157, -12.5742092, -13.7974424, -12.5833435, -0.5166574, 0.5514884
8: -5.8831196, -4.9708033, -5.8701897, -4.9533358, -0.4629235, 0.4357307
9: -10.3968105, -8.9332657, -10.4070578, -8.9496384, -1.0216455, 1.0468798

Time for backsubstitution: 8.22 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A1_B2_B1

### Relational analysis result of NS_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3762336, upper bound: 0.3780660
time: 5.32 seconds

## Relational analysis of NS_A1_B2_B2

### Relational analysis result of NS_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3788568, upper bound: 0.3786814
time: 3.35 seconds

## BFS NS instance: NS_A2_A1

### Backsubstitution after applying NS history:
0: -1.2873319, -0.1744130, -1.2973853, -0.1920335, -0.8226948, 0.8484211
1: -17.6885815, -16.2501221, -17.6853771, -16.2413292, -0.7732229, 0.7540259
2: -6.3101869, -4.9938669, -6.2657695, -4.9700098, -0.8607216, 0.8018422
3: -13.7330132, -12.5440063, -13.7171612, -12.5289593, -0.5564330, 0.5261264
4: -5.2193670, -4.0427089, -5.2506647, -4.0638514, -1.0585899, 1.1072464
5: -6.8065128, -5.9123716, -6.8078728, -5.9059973, -0.5202551, 0.5124646
6: 8.8205032, 9.8951883, 8.7896461, 9.8439083, -0.5467269, 0.6170785
7: -13.7966347, -12.5878000, -13.7724590, -12.5761080, -0.5502794, 0.5151823
8: -5.8531833, -4.9506035, -5.8729014, -4.9708109, -0.4353396, 0.4435486
9: -10.3880692, -8.9459410, -10.3885002, -8.9374599, -1.0184374, 1.0340772

Time for backsubstitution: 8.15 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.26 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of NS_A2_A1_B1

### Relational analysis result of NS_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3780659, upper bound: 0.3737706
time: 5.99 seconds

## Relational analysis of NS_A2_A1_B2

### Relational analysis result of NS_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3780661, upper bound: 0.3737708
time: 4.91 seconds

## BFS NS instance: NS_A2_A2

### Backsubstitution after applying NS history:
0: -1.2897872, -0.1785851, -1.2991303, -0.1920320, -0.8261418, 0.8501315
1: -17.6785374, -16.2289543, -17.6853771, -16.2304668, -0.7898636, 0.7641993
2: -6.3132534, -4.9937668, -6.2673302, -4.9699359, -0.8640308, 0.8039684
3: -13.7359457, -12.5258846, -13.7171612, -12.5207472, -0.5616882, 0.5322760
4: -5.2479272, -4.0368500, -5.2638121, -4.0626726, -1.0652370, 1.1218638
5: -6.8022747, -5.8991580, -6.8088045, -5.8989153, -0.5387919, 0.5104036
6: 8.8257513, 9.9162951, 8.7896423, 9.8552570, -0.5595808, 0.6226740
7: -13.7973042, -12.5843210, -13.7727814, -12.5743437, -0.5523143, 0.5192797
8: -5.8668628, -4.9533348, -5.8798380, -4.9708109, -0.4383848, 0.4558585
9: -10.4022627, -8.9496393, -10.3956156, -8.9374599, -1.0337930, 1.0403628

Time for backsubstitution: 8.03 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.24 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of NS_A2_A2_B1

### Relational analysis result of NS_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3786799, upper bound: 0.3760782
time: 4.13 seconds

## Relational analysis of NS_A2_A2_B2

### Relational analysis result of NS_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3786801, upper bound: 0.3760782
time: 3.55 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 15.95 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 15.95
Output dim: 6, lower bound: -0.3782938, upper bound: 0.3737713
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 15.95
Output dim: 6, lower bound: -0.3788568, upper bound: 0.3760782
NS_A1_B2_B1, status: Status.UNKNOWN, split count: 3, time: 15.95
Output dim: 6, lower bound: -0.3762336, upper bound: 0.3780660
NS_A1_B2_B2, status: Status.UNKNOWN, split count: 3, time: 15.95
Output dim: 6, lower bound: -0.3788568, upper bound: 0.3786814
NS_A2_A1_B1, status: Status.UNKNOWN, split count: 3, time: 15.95
Output dim: 6, lower bound: -0.3780659, upper bound: 0.3737706
NS_A2_A1_B2, status: Status.UNKNOWN, split count: 3, time: 15.95
Output dim: 6, lower bound: -0.3780661, upper bound: 0.3737708
NS_A2_A2_B1, status: Status.UNKNOWN, split count: 3, time: 15.95
Output dim: 6, lower bound: -0.3786799, upper bound: 0.3760782
NS_A2_A2_B2, status: Status.UNKNOWN, split count: 3, time: 15.95
Output dim: 6, lower bound: -0.3786801, upper bound: 0.3760782

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.2956977, -0.1877984, -1.2983388, -0.1919715, -0.8238912, 0.8307247
1: -17.6970196, -16.2564564, -17.6869736, -16.2412910, -0.7745826, 0.7531054
2: -6.2637243, -4.9665127, -6.2657685, -4.9663849, -0.7938824, 0.7957163
3: -13.7142849, -12.5409775, -13.7172222, -12.5301399, -0.5317912, 0.5236678
4: -5.2373524, -4.0678797, -5.2538257, -4.0638504, -1.0590134, 1.0718365
5: -6.8148837, -5.9151211, -6.8100758, -5.9059844, -0.5187590, 0.5117865
6: 8.7692194, 9.8289461, 8.7731266, 9.8439159, -0.5409120, 0.5260677
7: -13.7720079, -12.5785751, -13.7724609, -12.5761585, -0.5143182, 0.5120089
8: -5.8659511, -4.9680748, -5.8754025, -4.9708047, -0.4331686, 0.4410937
9: -10.3784018, -8.9295712, -10.3886185, -8.9332647, -1.0224466, 1.0369053

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.27 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3776640, upper bound: 0.3785128
time: 3.54 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3784391, upper bound: 0.3765145
time: 5.82 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.2981721, -0.1919711, -1.3000848, -0.1919723, -0.8273425, 0.8324065
1: -17.6869736, -16.2353001, -17.6869717, -16.2304306, -0.7912240, 0.7627716
2: -6.2667971, -4.9664106, -6.2673297, -4.9663110, -0.7971444, 0.7978430
3: -13.7172213, -12.5228529, -13.7172203, -12.5219326, -0.5370560, 0.5298181
4: -5.2646275, -4.0630426, -5.2664671, -4.0626712, -1.0656462, 1.0865035
5: -6.8107090, -5.9019070, -6.8110008, -5.8989019, -0.5371950, 0.5094918
6: 8.7731247, 9.8513508, 8.7731228, 9.8552675, -0.5537605, 0.5317738
7: -13.7726765, -12.5751715, -13.7727833, -12.5744314, -0.5164576, 0.5160930
8: -5.8797674, -4.9708042, -5.8823380, -4.9708037, -0.4362073, 0.4532012
9: -10.3921070, -8.9332657, -10.3957348, -8.9332666, -1.0374193, 1.0431919

Time for backsubstitution: 7.92 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of NS_A1_B1_A2_A1

### Relational analysis result of NS_A1_B1_A2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3814256, upper bound: 0.3787854
time: 4.06 seconds

## Relational analysis of NS_A1_B1_A2_A2

### Relational analysis result of NS_A1_B1_A2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3798805, upper bound: 0.3798818
time: 3.36 seconds

## BFS NS instance: NS_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1.2983388, -0.1919715, -1.2873319, -0.1744130, -0.8478117, 0.8197250
1: -17.6869736, -16.2412910, -17.6885815, -16.2501221, -0.7580049, 0.7636201
2: -6.2657685, -4.9663849, -6.3101869, -4.9938669, -0.7942195, 0.8579578
3: -13.7172222, -12.5301399, -13.7330132, -12.5440063, -0.5236471, 0.5548074
4: -5.2538257, -4.0638504, -5.2193670, -4.0427089, -1.1056414, 1.0521936
5: -6.8100758, -5.9059844, -6.8065128, -5.9123716, -0.5146987, 0.5115427
6: 8.7731266, 9.8439159, 8.8205032, 9.8951883, -0.6288471, 0.5224745
7: -13.7724609, -12.5761585, -13.7966347, -12.5878000, -0.5115533, 0.5483623
8: -5.8754025, -4.9708047, -5.8531833, -4.9506035, -0.4500866, 0.4151572
9: -10.3886185, -8.9332647, -10.3880692, -8.9459410, -1.0145593, 1.0248685

Time for backsubstitution: 8.60 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of NS_A1_B2_B1_A1

### Relational analysis result of NS_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3720268, upper bound: 0.3710599
time: 3.96 seconds

## Relational analysis of NS_A1_B2_B1_A2

### Relational analysis result of NS_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3707669, upper bound: 0.3725760
time: 5.17 seconds

## BFS NS instance: NS_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1.3000848, -0.1919723, -1.2897872, -0.1785851, -0.8495231, 0.8231916
1: -17.6869717, -16.2304306, -17.6785374, -16.2289543, -0.7678952, 0.7790649
2: -6.2673297, -4.9663110, -6.3132534, -4.9937668, -0.7963467, 0.8612661
3: -13.7172203, -12.5219326, -13.7359457, -12.5258846, -0.5297949, 0.5600643
4: -5.2664671, -4.0626712, -5.2479272, -4.0368500, -1.1202555, 1.0588269
5: -6.8110008, -5.8989019, -6.8022747, -5.8991580, -0.5124032, 0.5297642
6: 8.7731228, 9.8552675, 8.8257513, 9.9162951, -0.6344366, 0.5339887
7: -13.7727833, -12.5744314, -13.7973042, -12.5843210, -0.5153606, 0.5505021
8: -5.8823380, -4.9708037, -5.8668628, -4.9533348, -0.4623635, 0.4181799
9: -10.3957348, -8.9332666, -10.4022627, -8.9496393, -1.0208421, 1.0402222

Time for backsubstitution: 8.67 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of NS_A1_B2_B2_A1

### Relational analysis result of NS_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3746000, upper bound: 0.3715603
time: 3.75 seconds

## Relational analysis of NS_A1_B2_B2_A2

### Relational analysis result of NS_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3733884, upper bound: 0.3731784
time: 7.23 seconds

## BFS NS instance: NS_A2_A1_B1

### Backsubstitution after applying NS history:
0: -1.2873319, -0.1744130, -1.2983388, -0.1919715, -0.8197250, 0.8478117
1: -17.6885815, -16.2501221, -17.6869736, -16.2412910, -0.7636199, 0.7580049
2: -6.3101869, -4.9938669, -6.2657685, -4.9663849, -0.8579578, 0.7942190
3: -13.7330132, -12.5440063, -13.7172222, -12.5301399, -0.5548074, 0.5236468
4: -5.2193670, -4.0427089, -5.2538257, -4.0638504, -1.0521936, 1.1056414
5: -6.8065128, -5.9123716, -6.8100758, -5.9059844, -0.5115426, 0.5146985
6: 8.8205032, 9.8951883, 8.7731266, 9.8439159, -0.5224746, 0.6288471
7: -13.7966347, -12.5878000, -13.7724609, -12.5761585, -0.5483623, 0.5115533
8: -5.8531833, -4.9506035, -5.8754025, -4.9708047, -0.4151573, 0.4500866
9: -10.3880692, -8.9459410, -10.3886185, -8.9332647, -1.0248690, 1.0145593

Time for backsubstitution: 8.65 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of NS_A2_A1_B1_B1

### Relational analysis result of NS_A2_A1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3710589, upper bound: 0.3693555
time: 5.32 seconds

## Relational analysis of NS_A2_A1_B1_B2

### Relational analysis result of NS_A2_A1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3725763, upper bound: 0.3682842
time: 5.80 seconds

## BFS NS instance: NS_A2_A1_B2

### Backsubstitution after applying NS history:
0: -1.2873319, -0.1744130, -1.2899590, -0.1785848, -0.8244920, 0.8312907
1: -17.6885815, -16.2501221, -17.6785374, -16.2349586, -0.7735317, 0.7520542
2: -6.3101869, -4.9938669, -6.3122249, -4.9937391, -0.7988629, 0.8007250
3: -13.7330132, -12.5440063, -13.7359467, -12.5331707, -0.5349157, 0.5267866
4: -5.2193670, -4.0427089, -5.2363691, -4.0376515, -1.0559959, 1.0688195
5: -6.8065128, -5.9123716, -6.8016634, -5.9032359, -0.5200224, 0.5130578
6: 8.8205032, 9.8951883, 8.8257513, 9.9096422, -0.5468587, 0.5320208
7: -13.7966347, -12.5878000, -13.7970896, -12.5853310, -0.5174723, 0.5151484
8: -5.8531833, -4.9506035, -5.8625383, -4.9533339, -0.4352976, 0.4432276
9: -10.3880692, -8.9459410, -10.3986216, -8.9496384, -1.0202475, 1.0346956

Time for backsubstitution: 8.72 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of NS_A2_A1_B2_B1

### Relational analysis result of NS_A2_A1_B2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3710590, upper bound: 0.3693568
time: 3.95 seconds

## Relational analysis of NS_A2_A1_B2_B2

### Relational analysis result of NS_A2_A1_B2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3725765, upper bound: 0.3682842
time: 5.53 seconds

## BFS NS instance: NS_A2_A2_B1

### Backsubstitution after applying NS history:
0: -1.2897872, -0.1785851, -1.3000848, -0.1919723, -0.8231916, 0.8495235
1: -17.6785374, -16.2289543, -17.6869717, -16.2304306, -0.7790647, 0.7678952
2: -6.3132534, -4.9937668, -6.2673297, -4.9663110, -0.8612666, 0.7963467
3: -13.7359457, -12.5258846, -13.7172203, -12.5219326, -0.5600643, 0.5297952
4: -5.2479272, -4.0368500, -5.2664671, -4.0626712, -1.0588274, 1.1202550
5: -6.8022747, -5.8991580, -6.8110008, -5.8989019, -0.5297642, 0.5124034
6: 8.8257513, 9.9162951, 8.7731228, 9.8552675, -0.5339887, 0.6344366
7: -13.7973042, -12.5843210, -13.7727833, -12.5744314, -0.5505018, 0.5153606
8: -5.8668628, -4.9533348, -5.8823380, -4.9708037, -0.4181798, 0.4623636
9: -10.4022627, -8.9496393, -10.3957348, -8.9332666, -1.0402231, 1.0208416

Time for backsubstitution: 8.64 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 332

## Relational analysis of NS_A2_A2_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3715590, upper bound: 0.3716291
time: 4.30 seconds

## Relational analysis of NS_A2_A2_B1_B2

### Relational analysis result of NS_A2_A2_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3731785, upper bound: 0.3705888
time: 3.89 seconds

## BFS NS instance: NS_A2_A2_B2

### Backsubstitution after applying NS history:
0: -1.2897872, -0.1785851, -1.2916961, -0.1785840, -0.8279176, 0.8329968
1: -17.6785374, -16.2289543, -17.6785374, -16.2240868, -0.7901611, 0.7617176
2: -6.3132534, -4.9937668, -6.3137856, -4.9936628, -0.8021464, 0.8028660
3: -13.7359457, -12.5258846, -13.7359467, -12.5249634, -0.5401714, 0.5329432
4: -5.2479272, -4.0368500, -5.2495070, -4.0364828, -1.0626431, 1.0834866
5: -6.8022747, -5.8991580, -6.8025556, -5.8961535, -0.5385587, 0.5106928
6: 8.8257513, 9.9162951, 8.8257484, 9.9204988, -0.5597122, 0.5377178
7: -13.7973042, -12.5843210, -13.7974110, -12.5835676, -0.5196121, 0.5192463
8: -5.8668628, -4.9533348, -5.8694119, -4.9533358, -0.4383444, 0.4553337
9: -10.4022627, -8.9496393, -10.4059610, -8.9496393, -1.0352054, 1.0409842

Time for backsubstitution: 8.63 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of NS_A2_A2_B2_A1

### Relational analysis result of NS_A2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3740313, upper bound: 0.3691173
time: 3.95 seconds

## Relational analysis of NS_A2_A2_B2_A2

### Relational analysis result of NS_A2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3731786, upper bound: 0.3705888
time: 3.75 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 16.57 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3776640, upper bound: 0.3785128
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3784391, upper bound: 0.3765145
NS_A1_B1_A2_A1, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3814256, upper bound: 0.3787854
NS_A1_B1_A2_A2, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3798805, upper bound: 0.3798818
NS_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3720268, upper bound: 0.3710599
NS_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3707669, upper bound: 0.3725760
NS_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3746000, upper bound: 0.3715603
NS_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3733884, upper bound: 0.3731784
NS_A2_A1_B1_B1, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3710589, upper bound: 0.3693555
NS_A2_A1_B1_B2, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3725763, upper bound: 0.3682842
NS_A2_A1_B2_B1, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3710590, upper bound: 0.3693568
NS_A2_A1_B2_B2, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3725765, upper bound: 0.3682842
NS_A2_A2_B1_B1, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3715590, upper bound: 0.3716291
NS_A2_A2_B1_B2, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3731785, upper bound: 0.3705888
NS_A2_A2_B2_A1, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3740313, upper bound: 0.3691173
NS_A2_A2_B2_A2, status: Status.UNKNOWN, split count: 4, time: 16.57
Output dim: 6, lower bound: -0.3731786, upper bound: 0.3705888

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.2952582, -0.1908982, -1.3014746, -0.1982945, -0.8120151, 0.8252063
1: -17.6969719, -16.2683506, -17.6943474, -16.2673931, -0.7470305, 0.7412522
2: -6.2627058, -4.9672551, -6.2635784, -4.9618797, -0.7925398, 0.7889190
3: -13.7138548, -12.5440683, -13.7244101, -12.5367966, -0.5218377, 0.5215819
4: -5.2178564, -4.0685368, -5.2115703, -4.0692086, -1.0327768, 1.0355258
5: -6.8140545, -5.9181247, -6.8200684, -5.9123907, -0.5046952, 0.5096638
6: 8.7756586, 9.8289032, 8.7853680, 9.8561268, -0.5255336, 0.5021470
7: -13.7716112, -12.5787964, -13.7716475, -12.5765266, -0.5134356, 0.5107601
8: -5.8643279, -4.9753032, -5.8737149, -4.9862294, -0.4160534, 0.4296453
9: -10.3720741, -8.9299660, -10.3753424, -8.9364843, -1.0133309, 1.0238891

Time for backsubstitution: 8.77 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3696401, upper bound: 0.3734315
time: 5.48 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3696401, upper bound: 0.3695779
time: 7.19 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.2955616, -0.1893910, -1.2979411, -0.1965383, -0.8165951, 0.8256617
1: -17.6969604, -16.2608337, -17.6868038, -16.2531376, -0.7588444, 0.7480381
2: -6.2618113, -4.9668226, -6.2602167, -4.9672132, -0.7922955, 0.7917738
3: -13.7141113, -12.5432777, -13.7167625, -12.5368156, -0.5236173, 0.5218837
4: -5.2358012, -4.0680513, -5.2497768, -4.0643606, -1.0487957, 1.0464787
5: -6.8146086, -5.9184246, -6.8093185, -5.9154816, -0.5090051, 0.5095080
6: 8.7740211, 9.8289299, 8.7859993, 9.8438644, -0.5378866, 0.5038252
7: -13.7718639, -12.5786314, -13.7720442, -12.5763159, -0.5136609, 0.5125186
8: -5.8655238, -4.9694653, -5.8742104, -4.9748416, -0.4222238, 0.4370469
9: -10.3777828, -8.9296713, -10.3869028, -8.9335518, -1.0147705, 1.0382390

Time for backsubstitution: 8.65 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 75

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3699349, upper bound: 0.3711857
time: 3.70 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3699349, upper bound: 0.3681538
time: 4.79 seconds

## BFS NS instance: NS_A1_B1_A2_A1

### Backsubstitution after applying NS history:
0: -1.3013088, -0.1982956, -1.2996430, -0.1950695, -0.8217344, 0.8205118
1: -17.6943474, -16.2613850, -17.6869278, -16.2422523, -0.7799451, 0.7353492
2: -6.2645798, -4.9619060, -6.2662172, -4.9670544, -0.7903483, 0.7965012
3: -13.7244120, -12.5295038, -13.7167883, -12.5250225, -0.5349743, 0.5198619
4: -5.2223525, -4.0684109, -5.2469530, -4.0633698, -1.0293350, 1.0602756
5: -6.8207073, -5.9083061, -6.8101540, -5.9019060, -0.5350947, 0.4954140
6: 8.7853661, 9.8635740, 8.7795334, 9.8552208, -0.5298388, 0.5164232
7: -13.7718601, -12.5755415, -13.7723875, -12.5746508, -0.5152092, 0.5152135
8: -5.8780880, -4.9862289, -5.8807278, -4.9780345, -0.4246088, 0.4362185
9: -10.3788338, -8.9364843, -10.3894100, -8.9336567, -1.0244102, 1.0339308

Time for backsubstitution: 8.68 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A1_B1_A2_A1_B1

### Relational analysis result of NS_A1_B1_A2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3785093, upper bound: 0.3776652
time: 5.00 seconds

## Relational analysis of NS_A1_B1_A2_A1_B2

### Relational analysis result of NS_A1_B1_A2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3785093, upper bound: 0.3787837
time: 3.80 seconds

## BFS NS instance: NS_A1_B1_A2_A2

### Backsubstitution after applying NS history:
0: -1.2977750, -0.1965378, -1.2999457, -0.1935619, -0.8222585, 0.8251729
1: -17.6868038, -16.2470608, -17.6869164, -16.2346802, -0.7862074, 0.7471797
2: -6.2612448, -4.9672394, -6.2654181, -4.9666214, -0.7931995, 0.7962470
3: -13.7167625, -12.5295258, -13.7170477, -12.5242329, -0.5352724, 0.5216447
4: -5.2605710, -4.0635538, -5.2649064, -4.0628514, -1.0402508, 1.0762858
5: -6.8099527, -5.9114041, -6.8107214, -5.9022036, -0.5349283, 0.4997618
6: 8.7859974, 9.8512964, 8.7779160, 9.8552465, -0.5315841, 0.5287488
7: -13.7722588, -12.5753288, -13.7726412, -12.5744867, -0.5169382, 0.5154393
8: -5.8785849, -4.9748402, -5.8819141, -4.9721942, -0.4321042, 0.4424998
9: -10.3903923, -8.9335518, -10.3951187, -8.9333668, -1.0387139, 1.0357003

Time for backsubstitution: 8.81 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 75

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of NS_A1_B1_A2_A2_B1

### Relational analysis result of NS_A1_B1_A2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3740860, upper bound: 0.3708572
time: 3.88 seconds

## Relational analysis of NS_A1_B1_A2_A2_B2

### Relational analysis result of NS_A1_B1_A2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3708559, upper bound: 0.3708556
time: 4.83 seconds

## BFS NS instance: NS_A1_B2_B1_A1

### Backsubstitution after applying NS history:
0: -1.3014746, -0.1982945, -1.2868850, -0.1775031, -0.8423119, 0.8078380
1: -17.6943474, -16.2673931, -17.6885319, -16.2620106, -0.7461615, 0.7360635
2: -6.2635784, -4.9618797, -6.3091636, -4.9945917, -0.7874269, 0.8566160
3: -13.7244101, -12.5367966, -13.7325850, -12.5470982, -0.5215557, 0.5448427
4: -5.2115703, -4.0692086, -5.1999159, -4.0433788, -1.0693417, 1.0259824
5: -6.8200684, -5.9123907, -6.8057051, -5.9153771, -0.5125821, 0.4975283
6: 8.7853680, 9.8561268, 8.8268661, 9.8951445, -0.6049256, 0.5074265
7: -13.7716475, -12.5765266, -13.7962418, -12.5880156, -0.5103528, 0.5474806
8: -5.8737149, -4.9862294, -5.8516335, -4.9578314, -0.4386395, 0.3980861
9: -10.3753424, -8.9364843, -10.3817873, -8.9463177, -1.0015478, 1.0157743

Time for backsubstitution: 8.66 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.21 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of NS_A1_B2_B1_A1_B1

### Relational analysis result of NS_A1_B2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3663657, upper bound: 0.3611703
time: 5.70 seconds

## Relational analysis of NS_A1_B2_B1_A1_B2

### Relational analysis result of NS_A1_B2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3620371, upper bound: 0.3611706
time: 4.30 seconds

## BFS NS instance: NS_A1_B2_B1_A2

### Backsubstitution after applying NS history:
0: -1.2979411, -0.1965383, -1.2871927, -0.1760002, -0.8427472, 0.8124247
1: -17.6868038, -16.2531376, -17.6885262, -16.2545013, -0.7529252, 0.7478766
2: -6.2602167, -4.9672132, -6.3082738, -4.9941683, -0.7902746, 0.8563709
3: -13.7167625, -12.5368156, -13.7328405, -12.5463076, -0.5218606, 0.5466471
4: -5.2497768, -4.0643606, -5.2178097, -4.0428829, -1.0802860, 1.0419836
5: -6.8093185, -5.9154816, -6.8062449, -5.9156694, -0.5124223, 0.5017970
6: 8.7859993, 9.8438644, 8.8252869, 9.8951683, -0.6066024, 0.5195179
7: -13.7720442, -12.5763159, -13.7964935, -12.5878563, -0.5120754, 0.5477064
8: -5.8742104, -4.9748416, -5.8527746, -4.9519939, -0.4460409, 0.4042324
9: -10.3869028, -8.9335518, -10.3874626, -8.9460411, -1.0158944, 1.0171881

Time for backsubstitution: 8.64 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 75

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of NS_A1_B2_B1_A2_B1

### Relational analysis result of NS_A1_B2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3645002, upper bound: 0.3617731
time: 3.86 seconds

## Relational analysis of NS_A1_B2_B1_A2_B2

### Relational analysis result of NS_A1_B2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3610391, upper bound: 0.3617721
time: 8.16 seconds

## BFS NS instance: NS_A1_B2_B2_A1

### Backsubstitution after applying NS history:
0: -1.3032165, -0.1982946, -1.2893398, -0.1816746, -0.8440456, 0.8112106
1: -17.6943493, -16.2565098, -17.6784878, -16.2407837, -0.7564712, 0.7518961
2: -6.2651019, -4.9618049, -6.3121486, -4.9944887, -0.7895546, 0.8599248
3: -13.7244129, -12.5285845, -13.7355118, -12.5289745, -0.5277450, 0.5500963
4: -5.2241917, -4.0680456, -5.2284575, -4.0375586, -1.0839767, 1.0326028
5: -6.8210015, -5.9053082, -6.8014431, -5.9021606, -0.5103347, 0.5157509
6: 8.7853661, 9.8674965, 8.8320818, 9.9162483, -0.6105256, 0.5190001
7: -13.7719707, -12.5747986, -13.7969084, -12.5845394, -0.5141640, 0.5496194
8: -5.8806357, -4.9862289, -5.8653331, -4.9605637, -0.4510379, 0.4011259
9: -10.3824615, -8.9364853, -10.3959808, -8.9500103, -1.0078340, 1.0310035

Time for backsubstitution: 8.69 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of NS_A1_B2_B2_A1_A1

### Relational analysis result of NS_A1_B2_B2_A1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3720268, upper bound: 0.3696200
time: 5.05 seconds

## Relational analysis of NS_A1_B2_B2_A1_A2

### Relational analysis result of NS_A1_B2_B2_A1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3720268, upper bound: 0.3715603
time: 4.04 seconds

## BFS NS instance: NS_A1_B2_B2_A2

### Backsubstitution after applying NS history:
0: -1.2996868, -0.1965373, -1.2896476, -0.1801721, -0.8444662, 0.8159170
1: -17.6868038, -16.2421532, -17.6784782, -16.2332172, -0.7628317, 0.7636082
2: -6.2617784, -4.9671412, -6.3113394, -4.9940672, -0.7923980, 0.8596711
3: -13.7167616, -12.5286055, -13.7357740, -12.5281839, -0.5280230, 0.5519009
4: -5.2624111, -4.0631814, -5.2463632, -4.0370350, -1.0948706, 1.0486088
5: -6.8102469, -5.9083991, -6.8020010, -5.9024558, -0.5101471, 0.5200224
6: 8.7859974, 9.8552151, 8.8305254, 9.9162760, -0.6122384, 0.5310354
7: -13.7723675, -12.5745888, -13.7971601, -12.5843782, -0.5158582, 0.5498497
8: -5.8811464, -4.9748392, -5.8664589, -4.9547257, -0.4583437, 0.4074414
9: -10.3940201, -8.9335527, -10.4016552, -8.9497347, -1.0221171, 1.0327144

Time for backsubstitution: 8.67 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 75

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of NS_A1_B2_B2_A2_B1

### Relational analysis result of NS_A1_B2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3669789, upper bound: 0.3621035
time: 3.73 seconds

## Relational analysis of NS_A1_B2_B2_A2_B2

### Relational analysis result of NS_A1_B2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3634349, upper bound: 0.3621035
time: 3.65 seconds

## BFS NS instance: NS_A2_A1_B1_B1

### Backsubstitution after applying NS history:
0: -1.2868850, -0.1775031, -1.3014746, -0.1982945, -0.8078380, 0.8423119
1: -17.6885319, -16.2620106, -17.6943474, -16.2673931, -0.7360637, 0.7461615
2: -6.3091636, -4.9945917, -6.2635784, -4.9618797, -0.8566158, 0.7874269
3: -13.7325850, -12.5470982, -13.7244101, -12.5367966, -0.5448427, 0.5215557
4: -5.1999159, -4.0433788, -5.2115703, -4.0692086, -1.0259829, 1.0693421
5: -6.8057051, -5.9153771, -6.8200684, -5.9123907, -0.4975283, 0.5125821
6: 8.8268661, 9.8951445, 8.7853680, 9.8561268, -0.5074266, 0.6049256
7: -13.7962418, -12.5880156, -13.7716475, -12.5765266, -0.5474803, 0.5103528
8: -5.8516335, -4.9578314, -5.8737149, -4.9862294, -0.3980862, 0.4386395
9: -10.3817873, -8.9463177, -10.3753424, -8.9364843, -1.0157743, 1.0015478

Time for backsubstitution: 8.63 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of NS_A2_A1_B1_B1_A1

### Relational analysis result of NS_A2_A1_B1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3611705, upper bound: 0.3663669
time: 3.77 seconds

## Relational analysis of NS_A2_A1_B1_B1_A2

### Relational analysis result of NS_A2_A1_B1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3611705, upper bound: 0.3620368
time: 5.70 seconds

## BFS NS instance: NS_A2_A1_B1_B2

### Backsubstitution after applying NS history:
0: -1.2871927, -0.1760002, -1.2979411, -0.1965383, -0.8124247, 0.8427472
1: -17.6885262, -16.2545013, -17.6868038, -16.2531376, -0.7478766, 0.7529252
2: -6.3082738, -4.9941683, -6.2602167, -4.9672132, -0.8563709, 0.7902746
3: -13.7328405, -12.5463076, -13.7167625, -12.5368156, -0.5466471, 0.5218606
4: -5.2178097, -4.0428829, -5.2497768, -4.0643606, -1.0419836, 1.0802865
5: -6.8062449, -5.9156694, -6.8093185, -5.9154816, -0.5017970, 0.5124223
6: 8.8252869, 9.8951683, 8.7859993, 9.8438644, -0.5195179, 0.6066024
7: -13.7964935, -12.5878563, -13.7720442, -12.5763159, -0.5477066, 0.5120754
8: -5.8527746, -4.9519939, -5.8742104, -4.9748416, -0.4042323, 0.4460409
9: -10.3874626, -8.9460411, -10.3869028, -8.9335518, -1.0171881, 1.0158944

Time for backsubstitution: 8.71 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 75

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of NS_A2_A1_B1_B2_A1

### Relational analysis result of NS_A2_A1_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3617720, upper bound: 0.3645014
time: 3.49 seconds

## Relational analysis of NS_A2_A1_B1_B2_A2

### Relational analysis result of NS_A2_A1_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3617720, upper bound: 0.3610403
time: 3.45 seconds

## BFS NS instance: NS_A2_A1_B2_B1

### Backsubstitution after applying NS history:
0: -1.2868850, -0.1775031, -1.2930579, -0.1848934, -0.8126884, 0.8257928
1: -17.6885319, -16.2620106, -17.6859131, -16.2610512, -0.7460239, 0.7402313
2: -6.3091636, -4.9945917, -6.3100328, -4.9891500, -0.7975311, 0.7939234
3: -13.7325850, -12.5470982, -13.7431602, -12.5398273, -0.5249641, 0.5247226
4: -5.1999159, -4.0433788, -5.1941981, -4.0430269, -1.0297632, 1.0325270
5: -6.8057051, -5.9153771, -6.8117008, -5.9096427, -0.5059752, 0.5109295
6: 8.8268661, 9.8951445, 8.8379097, 9.9218569, -0.5314667, 0.5080798
7: -13.7962418, -12.5880156, -13.7962799, -12.5856905, -0.5165887, 0.5139015
8: -5.8516335, -4.9578314, -5.8610072, -4.9687586, -0.4181824, 0.4317857
9: -10.3817873, -8.9463177, -10.3854504, -8.9528074, -1.0111351, 1.0216565

Time for backsubstitution: 8.69 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 332
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.21 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of NS_A2_A1_B2_B1_A1

### Relational analysis result of NS_A2_A1_B2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3611705, upper bound: 0.3631356
time: 6.08 seconds

## Relational analysis of NS_A2_A1_B2_B1_A2

### Relational analysis result of NS_A2_A1_B2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3611706, upper bound: 0.3585766
time: 5.39 seconds

## BFS NS instance: NS_A2_A1_B2_B2

### Backsubstitution after applying NS history:
0: -1.2871927, -0.1760002, -1.2895542, -0.1831353, -0.8172216, 0.8262277
1: -17.6885262, -16.2545013, -17.6783695, -16.2467861, -0.7578449, 0.7469785
2: -6.3082738, -4.9941683, -6.3066750, -4.9945450, -0.7972705, 0.7967815
3: -13.7328405, -12.5463076, -13.7354860, -12.5398474, -0.5267603, 0.5250027
4: -5.2178097, -4.0428829, -5.2323246, -4.0381656, -1.0457745, 1.0434637
5: -6.8062449, -5.9156694, -6.8009233, -5.9127297, -0.5102696, 0.5108012
6: 8.8252869, 9.8951683, 8.8385820, 9.9095926, -0.5438340, 0.5097628
7: -13.7964935, -12.5878563, -13.7966757, -12.5854855, -0.5168142, 0.5156589
8: -5.8527746, -4.9519939, -5.8614006, -4.9573712, -0.4243613, 0.4391832
9: -10.3874626, -8.9460411, -10.3969393, -8.9499102, -1.0125570, 1.0360336

Time for backsubstitution: 8.67 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 75

Time for candidate selection: 0.22 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of NS_A2_A1_B2_B2_A1

### Relational analysis result of NS_A2_A1_B2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3617721, upper bound: 0.3616724
time: 4.19 seconds

## Relational analysis of NS_A2_A1_B2_B2_A2

### Relational analysis result of NS_A2_A1_B2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3617720, upper bound: 0.3579363
time: 4.48 seconds

## BFS NS instance: NS_A2_A2_B1_B1

### Backsubstitution after applying NS history:
0: -1.2893398, -0.1816746, -1.3032165, -0.1982946, -0.8112106, 0.8440456
1: -17.6784878, -16.2407837, -17.6943493, -16.2565098, -0.7518961, 0.7564712
2: -6.3121486, -4.9944887, -6.2651019, -4.9618049, -0.8599250, 0.7895546
3: -13.7355118, -12.5289745, -13.7244129, -12.5285845, -0.5500963, 0.5277450
4: -5.2284575, -4.0375586, -5.2241917, -4.0680456, -1.0326028, 1.0839767
5: -6.8014431, -5.9021606, -6.8210015, -5.9053082, -0.5157506, 0.5103346
6: 8.8320818, 9.9162483, 8.7853661, 9.8674965, -0.5189999, 0.6105256
7: -13.7969084, -12.5845394, -13.7719707, -12.5747986, -0.5496194, 0.5141640
8: -5.8653331, -4.9605637, -5.8806357, -4.9862289, -0.4011259, 0.4510380
9: -10.3959808, -8.9500103, -10.3824615, -8.9364853, -1.0310040, 1.0078340

Time for backsubstitution: 8.69 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 151

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A2_A2_B1_B1_B1

### Relational analysis result of NS_A2_A2_B1_B1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3696189, upper bound: 0.3741328
time: 6.07 seconds

## Relational analysis of NS_A2_A2_B1_B1_B2

### Relational analysis result of NS_A2_A2_B1_B1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3696189, upper bound: 0.3745996
time: 4.58 seconds

## BFS NS instance: NS_A2_A2_B1_B2

### Backsubstitution after applying NS history:
0: -1.2896476, -0.1801721, -1.2996868, -0.1965373, -0.8159170, 0.8444662
1: -17.6784782, -16.2332172, -17.6868038, -16.2421532, -0.7636082, 0.7628317
2: -6.3113394, -4.9940672, -6.2617784, -4.9671412, -0.8596706, 0.7923985
3: -13.7357740, -12.5281839, -13.7167616, -12.5286055, -0.5519006, 0.5280230
4: -5.2463632, -4.0370350, -5.2624111, -4.0631814, -1.0486088, 1.0948706
5: -6.8020010, -5.9024558, -6.8102469, -5.9083991, -0.5200224, 0.5101471
6: 8.8305254, 9.9162760, 8.7859974, 9.8552151, -0.5310352, 0.6122382
7: -13.7971601, -12.5843782, -13.7723675, -12.5745888, -0.5498497, 0.5158582
8: -5.8664589, -4.9547257, -5.8811464, -4.9748392, -0.4074414, 0.4583436
9: -10.4016552, -8.9497347, -10.3940201, -8.9335527, -1.0327144, 1.0221176

Time for backsubstitution: 8.86 seconds

### NS candidates at layer 3
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 76
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 890
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 613
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 1506
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 192
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 330
type: A, layer: 3, pos: 738
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 1406
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 75

Time for candidate selection: 0.26 seconds

### Candidate
type: A, layer: 3, pos: 1502

## Relational analysis of NS_A2_A2_B1_B2_A1

### Relational analysis result of NS_A2_A2_B1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3621023, upper bound: 0.3669801
time: 4.19 seconds

## Relational analysis of NS_A2_A2_B1_B2_A2

### Relational analysis result of NS_A2_A2_B1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3621023, upper bound: 0.3634363
time: 3.65 seconds

## BFS NS instance: NS_A2_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.2928877, -0.1848931, -1.2912495, -0.1816728, -0.8223295, 0.8211775
1: -17.6859131, -16.2550354, -17.6784859, -16.2359047, -0.7789125, 0.7343402
2: -6.3110347, -4.9891758, -6.3126678, -4.9943895, -0.7953446, 0.8015361
3: -13.7431574, -12.5325346, -13.7355118, -12.5280561, -0.5381122, 0.5229883
4: -5.2057357, -4.0422363, -5.2300386, -4.0371895, -1.0263529, 1.0572658
5: -6.8123112, -5.9055572, -6.8017249, -5.8991566, -0.5364442, 0.4966097
6: 8.8379068, 9.9285192, 8.8320808, 9.9204540, -0.5357709, 0.5223529
7: -13.7964954, -12.5846853, -13.7970181, -12.5837841, -0.5183635, 0.5183659
8: -5.8653440, -4.9687576, -5.8678741, -4.9605618, -0.4267533, 0.4383521
9: -10.3890915, -8.9528074, -10.3996782, -8.9500093, -1.0221720, 1.0317273

Time for backsubstitution: 8.72 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 1502
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 1705
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 660
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 192
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 75
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 1406
type: A, layer: 3, pos: 151
type: B, layer: 3, pos: 151

Time for candidate selection: 0.22 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A2_A2_B2_A1_B1

### Relational analysis result of NS_A2_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3721270, upper bound: 0.3685844
time: 4.83 seconds

## Relational analysis of NS_A2_A2_B2_A1_B2

### Relational analysis result of NS_A2_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3721270, upper bound: 0.3691157
time: 4.29 seconds

## BFS NS instance: NS_A2_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.2893839, -0.1831356, -1.2915574, -0.1801711, -0.8228340, 0.8257909
1: -17.6783695, -16.2407036, -17.6784782, -16.2283363, -0.7851367, 0.7461786
2: -6.3077011, -4.9945741, -6.3118730, -4.9939651, -0.7982008, 0.8012657
3: -13.7354851, -12.5325565, -13.7357712, -12.5272646, -0.5383880, 0.5247883
4: -5.2438745, -4.0373678, -5.2479429, -4.0366664, -1.0372515, 1.0732679
5: -6.8015342, -5.9086514, -6.8022823, -5.8994503, -0.5363021, 0.5009362
6: 8.8385820, 9.9162426, 8.8305225, 9.9204817, -0.5375208, 0.5346935
7: -13.7968884, -12.5844793, -13.7972689, -12.5836229, -0.5200922, 0.5185924
8: -5.8657303, -4.9573708, -5.8690052, -4.9547257, -0.4342426, 0.4446378
9: -10.4005804, -8.9499092, -10.4053574, -8.9497356, -1.0365009, 1.0334907

Time for backsubstitution: 9.11 seconds

### NS candidates at layer 3
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 1479
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 2480
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2901
type: A, layer: 3, pos: 2901
type: B, layer: 3, pos: 1979
type: A, layer: 3, pos: 1979
type: B, layer: 3, pos: 76
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 75
type: B, layer: 3, pos: 1775
type: A, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 613
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 1506
type: A, layer: 3, pos: 1515
type: B, layer: 3, pos: 2858
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 2858
type: B, layer: 3, pos: 1515
type: A, layer: 3, pos: 1228
type: B, layer: 3, pos: 1228
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 1847
type: B, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: B, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 38
type: A, layer: 3, pos: 330
type: B, layer: 3, pos: 738
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 1406
type: B, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: A, layer: 3, pos: 1406
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 75

Time for candidate selection: 0.25 seconds

### Candidate
type: B, layer: 3, pos: 1502

## Relational analysis of NS_A2_A2_B2_A2_B1

### Relational analysis result of NS_A2_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3659496, upper bound: 0.3600986
time: 3.51 seconds

## Relational analysis of NS_A2_A2_B2_A2_B2

### Relational analysis result of NS_A2_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3621022, upper bound: 0.3600974
time: 5.18 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 18.06 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3696401, upper bound: 0.3734315
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3696401, upper bound: 0.3695779
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3699349, upper bound: 0.3711857
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3699349, upper bound: 0.3681538
NS_A1_B1_A2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3785093, upper bound: 0.3776652
NS_A1_B1_A2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3785093, upper bound: 0.3787837
NS_A1_B1_A2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3740860, upper bound: 0.3708572
NS_A1_B1_A2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3708559, upper bound: 0.3708556
NS_A1_B2_B1_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3663657, upper bound: 0.3611703
NS_A1_B2_B1_A1_B2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3620371, upper bound: 0.3611706
NS_A1_B2_B1_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3645002, upper bound: 0.3617731
NS_A1_B2_B1_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3610391, upper bound: 0.3617721
NS_A1_B2_B2_A1_A1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3720268, upper bound: 0.3696200
NS_A1_B2_B2_A1_A2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3720268, upper bound: 0.3715603
NS_A1_B2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3669789, upper bound: 0.3621035
NS_A1_B2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3634349, upper bound: 0.3621035
NS_A2_A1_B1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3611705, upper bound: 0.3663669
NS_A2_A1_B1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3611705, upper bound: 0.3620368
NS_A2_A1_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3617720, upper bound: 0.3645014
NS_A2_A1_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3617720, upper bound: 0.3610403
NS_A2_A1_B2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3611705, upper bound: 0.3631356
NS_A2_A1_B2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3611706, upper bound: 0.3585766
NS_A2_A1_B2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3617721, upper bound: 0.3616724
NS_A2_A1_B2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3617720, upper bound: 0.3579363
NS_A2_A2_B1_B1_B1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3696189, upper bound: 0.3741328
NS_A2_A2_B1_B1_B2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3696189, upper bound: 0.3745996
NS_A2_A2_B1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3621023, upper bound: 0.3669801
NS_A2_A2_B1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3621023, upper bound: 0.3634363
NS_A2_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3721270, upper bound: 0.3685844
NS_A2_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3721270, upper bound: 0.3691157
NS_A2_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3659496, upper bound: 0.3600986
NS_A2_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 5, time: 18.06
Output dim: 6, lower bound: -0.3621022, upper bound: 0.3600974

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.84 + 550.09 = 607.93 seconds
