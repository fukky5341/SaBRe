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
execution time: IAR + RelationalAnalysis = 22.88 + 35.03 = 57.91 seconds
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
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.40 seconds

### Candidate
type: A, layer: 3, pos: 2130

## Relational analysis of NS_A1

### Relational analysis result of NS_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3828351, upper bound: 0.3830028
time: 3.45 seconds

## Relational analysis of NS_A2

### Relational analysis result of NS_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3830011, upper bound: 0.3830028
time: 3.40 seconds

## Summary of splitting at layer (split count: 0)
- Time for NS candidates: 7.27 seconds
NS_A1, status: Status.UNKNOWN, split count: 1, time: 7.27
Output dim: 6, lower bound: -0.3828351, upper bound: 0.3830028
NS_A2, status: Status.UNKNOWN, split count: 1, time: 7.27
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

Time for backsubstitution: 8.70 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406

Time for candidate selection: 0.50 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of NS_A1_B1

### Relational analysis result of NS_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3802265, upper bound: 0.3802265
time: 6.28 seconds

## Relational analysis of NS_A1_B2

### Relational analysis result of NS_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3802265, upper bound: 0.3830015
time: 4.63 seconds

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

Time for backsubstitution: 8.59 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 2130
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 2130

## Relational analysis of NS_A2_B1

### Relational analysis result of NS_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3830014, upper bound: 0.3802265
time: 5.69 seconds

## Relational analysis of NS_A2_B2

### Relational analysis result of NS_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3830014, upper bound: 0.3830016
time: 4.55 seconds

## Summary of splitting at layer (split count: 1)
- Time for NS candidates: 19.24 seconds
NS_A1_B1, status: Status.UNKNOWN, split count: 2, time: 19.24
Output dim: 6, lower bound: -0.3802265, upper bound: 0.3802265
NS_A1_B2, status: Status.UNKNOWN, split count: 2, time: 19.24
Output dim: 6, lower bound: -0.3802265, upper bound: 0.3830015
NS_A2_B1, status: Status.UNKNOWN, split count: 2, time: 19.24
Output dim: 6, lower bound: -0.3830014, upper bound: 0.3802265
NS_A2_B2, status: Status.UNKNOWN, split count: 2, time: 19.24
Output dim: 6, lower bound: -0.3830014, upper bound: 0.3830016

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

Time for backsubstitution: 8.61 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.44 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of NS_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3785822, upper bound: 0.3733013
time: 3.60 seconds

## Relational analysis of NS_A1_B1_A2

### Relational analysis result of NS_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3773689, upper bound: 0.3747377
time: 4.77 seconds

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

Time for backsubstitution: 8.59 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.43 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of NS_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3785822, upper bound: 0.3759035
time: 3.57 seconds

## Relational analysis of NS_A1_B2_A2

### Relational analysis result of NS_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3773689, upper bound: 0.3775125
time: 3.38 seconds

## BFS NS instance: NS_A2_B1

### Backsubstitution after applying NS history:
0: -1.2922771, -0.1785847, -1.3006644, -0.1919719, -0.8285227, 0.8497963
1: -17.6785374, -16.2226295, -17.6869717, -16.2289658, -0.7796826, 0.7960620
2: -6.3139429, -4.9936342, -6.2674885, -4.9662809, -0.8620272, 0.7963719
3: -13.7359467, -12.5246744, -13.7172203, -12.5216417, -0.5604260, 0.5373929
4: -5.2499752, -4.0363760, -5.2670107, -4.0625620, -1.0814323, 1.1219883
5: -6.8026385, -5.8952284, -6.8110862, -5.8979774, -0.5309906, 0.5406417
6: 8.8257484, 9.9217463, 8.7731228, 9.8564253, -0.5345092, 0.6559501
7: -13.7974424, -12.5833435, -13.7728157, -12.5742092, -0.5514884, 0.5166574
8: -5.8701897, -4.9533358, -5.8831196, -4.9708033, -0.4357307, 0.4629236
9: -10.4070578, -8.9496384, -10.3968105, -8.9332657, -1.0468798, 1.0216455

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.40 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of NS_A2_B1_A1

### Relational analysis result of NS_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3784122, upper bound: 0.3733002
time: 4.33 seconds

## Relational analysis of NS_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3775108, upper bound: 0.3747377
time: 3.68 seconds

## BFS NS instance: NS_A2_B2

### Backsubstitution after applying NS history:
0: -1.2922771, -0.1785847, -1.2922771, -0.1785847, -0.8332725, 0.8332725
1: -17.6785374, -16.2226295, -17.6785374, -16.2226295, -0.7907813, 0.7907810
2: -6.3139429, -4.9936342, -6.3139429, -4.9936342, -0.8028772, 0.8028769
3: -13.7359467, -12.5246744, -13.7359467, -12.5246744, -0.5405335, 0.5405335
4: -5.2499752, -4.0363760, -5.2499752, -4.0363760, -1.0852270, 1.0852270
5: -6.8026385, -5.8952284, -6.8026385, -5.8952284, -0.5398102, 0.5398102
6: 8.8257484, 9.9217463, 8.8257484, 9.9217463, -0.5602322, 0.5602324
7: -13.7974424, -12.5833435, -13.7974424, -12.5833435, -0.5206304, 0.5206301
8: -5.8701897, -4.9533358, -5.8701897, -4.9533358, -0.4558948, 0.4558948
9: -10.4070578, -8.9496384, -10.4070578, -8.9496384, -1.0417871, 1.0417876

Time for backsubstitution: 8.63 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 332
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.41 seconds

### Candidate
type: A, layer: 3, pos: 332

## Relational analysis of NS_A2_B2_A1

### Relational analysis result of NS_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3784123, upper bound: 0.3733003
time: 4.72 seconds

## Relational analysis of NS_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3775110, upper bound: 0.3747377
time: 3.56 seconds

## Summary of splitting at layer (split count: 2)
- Time for NS candidates: 17.33 seconds
NS_A1_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.33
Output dim: 6, lower bound: -0.3785822, upper bound: 0.3733013
NS_A1_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.33
Output dim: 6, lower bound: -0.3773689, upper bound: 0.3747377
NS_A1_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.33
Output dim: 6, lower bound: -0.3785822, upper bound: 0.3759035
NS_A1_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.33
Output dim: 6, lower bound: -0.3773689, upper bound: 0.3775125
NS_A2_B1_A1, status: Status.UNKNOWN, split count: 3, time: 17.33
Output dim: 6, lower bound: -0.3784122, upper bound: 0.3733002
NS_A2_B1_A2, status: Status.UNKNOWN, split count: 3, time: 17.33
Output dim: 6, lower bound: -0.3775108, upper bound: 0.3747377
NS_A2_B2_A1, status: Status.UNKNOWN, split count: 3, time: 17.33
Output dim: 6, lower bound: -0.3784123, upper bound: 0.3733003
NS_A2_B2_A2, status: Status.UNKNOWN, split count: 3, time: 17.33
Output dim: 6, lower bound: -0.3775110, upper bound: 0.3747377

## BFS NS instance: NS_A1_B1_A1

### Backsubstitution after applying NS history:
0: -1.3037968, -0.1982950, -1.3002234, -0.1950681, -0.8271837, 0.8207841
1: -17.6943493, -16.2550163, -17.6869240, -16.2407742, -0.7805719, 0.7646778
2: -6.2652569, -4.9617739, -6.2663717, -4.9670243, -0.7910719, 0.7965264
3: -13.7244110, -12.5282974, -13.7167864, -12.5247345, -0.5353370, 0.5274620
4: -5.2247367, -4.0679355, -5.2474976, -4.0632606, -1.0519552, 1.0620213
5: -6.8210869, -5.9043846, -6.8102403, -5.9009800, -0.5363483, 0.5243859
6: 8.7853661, 9.8686552, 8.7795334, 9.8563824, -0.5303597, 0.5389643
7: -13.7720013, -12.5745792, -13.7724190, -12.5744305, -0.5162234, 0.5165825
8: -5.8814173, -4.9862280, -5.8815117, -4.9780335, -0.4424062, 0.4367646
9: -10.3835392, -8.9364853, -10.3904877, -8.9336567, -1.0309858, 1.0347300

Time for backsubstitution: 8.67 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406

Time for candidate selection: 0.39 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A1_B1_A1_B1

### Relational analysis result of NS_A1_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3785115, upper bound: 0.3776638
time: 6.79 seconds

## Relational analysis of NS_A1_B1_A1_B2

### Relational analysis result of NS_A1_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3814255, upper bound: 0.3787855
time: 3.37 seconds

## BFS NS instance: NS_A1_B1_A2

### Backsubstitution after applying NS history:
0: -1.3002660, -0.1965364, -1.3005266, -0.1935626, -0.8276243, 0.8254490
1: -17.6868038, -16.2406693, -17.6869164, -16.2332039, -0.7868245, 0.7764060
2: -6.2619371, -4.9671116, -6.2655740, -4.9665923, -0.7939222, 0.7962723
3: -13.7167616, -12.5283175, -13.7170477, -12.5239439, -0.5356352, 0.5292411
4: -5.2629552, -4.0630717, -5.2654514, -4.0627432, -1.0628529, 1.0780272
5: -6.8103333, -5.9074755, -6.8108063, -5.9012804, -0.5361774, 0.5287147
6: 8.7859964, 9.8563747, 8.7779160, 9.8564062, -0.5321085, 0.5512595
7: -13.7723999, -12.5743704, -13.7726707, -12.5742655, -0.5179546, 0.5168197
8: -5.8819294, -4.9748406, -5.8826957, -4.9721937, -0.4497299, 0.4430622
9: -10.3950958, -8.9335537, -10.3961935, -8.9333658, -1.0452662, 1.0365133

Time for backsubstitution: 7.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A1_B1_A2_B1

### Relational analysis result of NS_A1_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3765146, upper bound: 0.3784390
time: 4.51 seconds

## Relational analysis of NS_A1_B1_A2_B2

### Relational analysis result of NS_A1_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3798804, upper bound: 0.3798819
time: 3.31 seconds

## BFS NS instance: NS_A1_B2_A1

### Backsubstitution after applying NS history:
0: -1.3037968, -0.1982950, -1.2918290, -0.1816724, -0.8443189, 0.8166151
1: -17.6943493, -16.2550163, -17.6784859, -16.2344303, -0.7847998, 0.7525134
2: -6.2652569, -4.9617739, -6.3128252, -4.9943590, -0.7895808, 0.8606873
3: -13.7244110, -12.5282974, -13.7355118, -12.5277662, -0.5353067, 0.5504591
4: -5.2247367, -4.0679355, -5.2305059, -4.0370808, -1.0857096, 1.0552359
5: -6.8210869, -5.9043846, -6.8018074, -5.8982325, -0.5385573, 0.5169780
6: 8.7853661, 9.8686552, 8.8320789, 9.9216995, -0.6320288, 0.5195246
7: -13.7720013, -12.5745792, -13.7970495, -12.5835590, -0.5154541, 0.5505960
8: -5.8814173, -4.9862280, -5.8686571, -4.9605641, -0.4515704, 0.4187790
9: -10.3835392, -8.9364853, -10.4007778, -8.9500103, -1.0086398, 1.0376339

Time for backsubstitution: 8.75 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A1_B2_A1_B1

### Relational analysis result of NS_A1_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3720268, upper bound: 0.3710599
time: 3.94 seconds

## Relational analysis of NS_A1_B2_A1_B2

### Relational analysis result of NS_A1_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3746000, upper bound: 0.3715603
time: 3.70 seconds

## BFS NS instance: NS_A1_B2_A2

### Backsubstitution after applying NS history:
0: -1.3002660, -0.1965364, -1.2921354, -0.1801710, -0.8447394, 0.8212886
1: -17.6868038, -16.2406693, -17.6784782, -16.2268677, -0.7910311, 0.7642415
2: -6.2619371, -4.9671116, -6.3120308, -4.9939370, -0.7924232, 0.8604326
3: -13.7167616, -12.5283175, -13.7357712, -12.5269737, -0.5356076, 0.5522630
4: -5.2629552, -4.0630717, -5.2484107, -4.0365562, -1.0966024, 1.0712252
5: -6.8103333, -5.9074755, -6.8023653, -5.8985267, -0.5383811, 0.5212470
6: 8.7859964, 9.8563747, 8.8305225, 9.9217243, -0.6337757, 0.5315564
7: -13.7723999, -12.5743704, -13.7973013, -12.5833988, -0.5171471, 0.5508332
8: -5.8819294, -4.9748406, -5.8697824, -4.9547253, -0.4588938, 0.4250480
9: -10.3950958, -8.9335537, -10.4064512, -8.9497337, -1.0229158, 1.0394030

Time for backsubstitution: 8.80 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A1_B2_A2_B1

### Relational analysis result of NS_A1_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3707669, upper bound: 0.3725760
time: 5.24 seconds

## Relational analysis of NS_A1_B2_A2_B2

### Relational analysis result of NS_A1_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3733884, upper bound: 0.3731784
time: 7.27 seconds

## BFS NS instance: NS_A2_B1_A1

### Backsubstitution after applying NS history:
0: -1.2953745, -0.1848918, -1.3002234, -0.1950681, -0.8229642, 0.8379655
1: -17.6859131, -16.2486744, -17.6869240, -16.2407742, -0.7683938, 0.7688947
2: -6.3117104, -4.9890461, -6.2663717, -4.9670243, -0.8552322, 0.7950535
3: -13.7431574, -12.5313292, -13.7167864, -12.5247345, -0.5583367, 0.5274048
4: -5.2077827, -4.0417662, -5.2474976, -4.0632606, -1.0452156, 1.0957885
5: -6.8126745, -5.9016347, -6.8102403, -5.9009800, -0.5289598, 0.5266283
6: 8.8379040, 9.9339762, 8.7795334, 9.8563824, -0.5110505, 0.6406302
7: -13.7966347, -12.5837021, -13.7724190, -12.5744305, -0.5502424, 0.5158727
8: -5.8686643, -4.9687586, -5.8815117, -4.9780335, -0.4244721, 0.4459338
9: -10.3938885, -8.9528065, -10.3904877, -8.9336567, -1.0338936, 1.0123858

Time for backsubstitution: 8.74 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406

Time for candidate selection: 0.40 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A2_B1_A1_B1

### Relational analysis result of NS_A2_B1_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3721365, upper bound: 0.3715071
time: 5.25 seconds

## Relational analysis of NS_A2_B1_A1_B2

### Relational analysis result of NS_A2_B1_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3740310, upper bound: 0.3720184
time: 3.94 seconds

## BFS NS instance: NS_A2_B1_A2

### Backsubstitution after applying NS history:
0: -1.2918727, -0.1831341, -1.3005266, -0.1935626, -0.8234577, 0.8425870
1: -17.6783695, -16.2343159, -17.6869164, -16.2332039, -0.7746494, 0.7806633
2: -6.3083930, -4.9944420, -6.2655740, -4.9665923, -0.8580825, 0.7947655
3: -13.7354879, -12.5313492, -13.7170477, -12.5239439, -0.5586326, 0.5292106
4: -5.2459226, -4.0368900, -5.2654514, -4.0627432, -1.0560694, 1.1117783
5: -6.8018985, -5.9047236, -6.8108063, -5.9012804, -0.5287607, 0.5309253
6: 8.8385801, 9.9216938, 8.7779160, 9.8564062, -0.5125241, 0.6529286
7: -13.7970314, -12.5834990, -13.7726707, -12.5742655, -0.5519702, 0.5160363
8: -5.8690519, -4.9573698, -5.8826957, -4.9721937, -0.4317322, 0.4522305
9: -10.4053764, -8.9499121, -10.3961935, -8.9333658, -1.0481787, 1.0141654

Time for backsubstitution: 8.83 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A2_B1_A2_B1

### Relational analysis result of NS_A2_B1_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3711652, upper bound: 0.3728278
time: 6.08 seconds

## Relational analysis of NS_A2_B1_A2_B2

### Relational analysis result of NS_A2_B1_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3731782, upper bound: 0.3733900
time: 3.75 seconds

## BFS NS instance: NS_A2_B2_A1

### Backsubstitution after applying NS history:
0: -1.2953745, -0.1848918, -1.2918290, -0.1816724, -0.8277969, 0.8214502
1: -17.6859131, -16.2486744, -17.6784859, -16.2344303, -0.7795408, 0.7636607
2: -6.3117104, -4.9890461, -6.3128252, -4.9943590, -0.7960746, 0.8015466
3: -13.7431574, -12.5313292, -13.7355118, -12.5277662, -0.5384748, 0.5305793
4: -5.2077827, -4.0417662, -5.2305059, -4.0370808, -1.0489597, 1.0590124
5: -6.8126745, -5.9016347, -6.8018074, -5.8982325, -0.5376997, 0.5257579
6: 8.8379040, 9.9339762, 8.8320789, 9.9216995, -0.5362904, 0.5449026
7: -13.7966347, -12.5837021, -13.7970495, -12.5835590, -0.5193779, 0.5197361
8: -5.8686643, -4.9687586, -5.8686571, -4.9605641, -0.4445467, 0.4388989
9: -10.3938885, -8.9528065, -10.4007778, -8.9500103, -1.0287552, 1.0325255

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A2_B2_A1_B1

### Relational analysis result of NS_A2_B2_A1_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3721365, upper bound: 0.3685845
time: 6.24 seconds

## Relational analysis of NS_A2_B2_A1_B2

### Relational analysis result of NS_A2_B2_A1_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3740310, upper bound: 0.3691173
time: 3.71 seconds

## BFS NS instance: NS_A2_B2_A2

### Backsubstitution after applying NS history:
0: -1.2918727, -0.1831341, -1.2921354, -0.1801710, -0.8282161, 0.8260665
1: -17.6783695, -16.2343159, -17.6784782, -16.2268677, -0.7857552, 0.7753966
2: -6.3083930, -4.9944420, -6.3120308, -4.9939370, -0.7989302, 0.8012753
3: -13.7354879, -12.5313492, -13.7357712, -12.5269737, -0.5387504, 0.5323756
4: -5.2459226, -4.0368900, -5.2484107, -4.0365562, -1.0598450, 1.0750113
5: -6.8018985, -5.9047236, -6.8023653, -5.8985267, -0.5375543, 0.5300574
6: 8.8385801, 9.9216938, 8.8305225, 9.9217243, -0.5380452, 0.5572116
7: -13.7970314, -12.5834990, -13.7973013, -12.5833988, -0.5211074, 0.5199742
8: -5.8690519, -4.9573698, -5.8697824, -4.9547253, -0.4518647, 0.4452010
9: -10.4053764, -8.9499121, -10.4064512, -8.9497337, -1.0430617, 1.0343013

Time for backsubstitution: 8.83 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: B, layer: 3, pos: 1479
type: B, layer: 3, pos: 2480
type: B, layer: 3, pos: 332
type: B, layer: 3, pos: 1705
type: B, layer: 3, pos: 1502
type: B, layer: 3, pos: 2818
type: B, layer: 3, pos: 555
type: B, layer: 3, pos: 2901
type: B, layer: 3, pos: 76
type: B, layer: 3, pos: 1979
type: B, layer: 3, pos: 1775
type: B, layer: 3, pos: 3102
type: B, layer: 3, pos: 613
type: B, layer: 3, pos: 890
type: B, layer: 3, pos: 780
type: B, layer: 3, pos: 660
type: B, layer: 3, pos: 2858
type: B, layer: 3, pos: 1228
type: B, layer: 3, pos: 1515
type: B, layer: 3, pos: 1847
type: B, layer: 3, pos: 2396
type: B, layer: 3, pos: 38
type: B, layer: 3, pos: 1506
type: B, layer: 3, pos: 75
type: B, layer: 3, pos: 738
type: B, layer: 3, pos: 2827
type: B, layer: 3, pos: 192
type: B, layer: 3, pos: 330
type: B, layer: 3, pos: 2356
type: B, layer: 3, pos: 151
type: B, layer: 3, pos: 1406

Time for candidate selection: 0.41 seconds

### Candidate
type: B, layer: 3, pos: 1479

## Relational analysis of NS_A2_B2_A2_B1

### Relational analysis result of NS_A2_B2_A2_B1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3711652, upper bound: 0.3699922
time: 4.04 seconds

## Relational analysis of NS_A2_B2_A2_B2

### Relational analysis result of NS_A2_B2_A2_B2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3731784, upper bound: 0.3705889
time: 3.69 seconds

## Summary of splitting at layer (split count: 3)
- Time for NS candidates: 16.98 seconds
NS_A1_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3785115, upper bound: 0.3776638
NS_A1_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3814255, upper bound: 0.3787855
NS_A1_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3765146, upper bound: 0.3784390
NS_A1_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3798804, upper bound: 0.3798819
NS_A1_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3720268, upper bound: 0.3710599
NS_A1_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3746000, upper bound: 0.3715603
NS_A1_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3707669, upper bound: 0.3725760
NS_A1_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3733884, upper bound: 0.3731784
NS_A2_B1_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3721365, upper bound: 0.3715071
NS_A2_B1_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3740310, upper bound: 0.3720184
NS_A2_B1_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3711652, upper bound: 0.3728278
NS_A2_B1_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3731782, upper bound: 0.3733900
NS_A2_B2_A1_B1, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3721365, upper bound: 0.3685845
NS_A2_B2_A1_B2, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3740310, upper bound: 0.3691173
NS_A2_B2_A2_B1, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3711652, upper bound: 0.3699922
NS_A2_B2_A2_B2, status: Status.UNKNOWN, split count: 4, time: 16.98
Output dim: 6, lower bound: -0.3731784, upper bound: 0.3705889

## BFS NS instance: NS_A1_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.3014746, -0.1982945, -1.2952582, -0.1908982, -0.8252063, 0.8120151
1: -17.6943474, -16.2673931, -17.6969719, -16.2683506, -0.7412522, 0.7470307
2: -6.2635784, -4.9618797, -6.2627058, -4.9672551, -0.7889190, 0.7925401
3: -13.7244101, -12.5367966, -13.7138548, -12.5440683, -0.5215819, 0.5218377
4: -5.2115703, -4.0692086, -5.2178564, -4.0685368, -1.0355263, 1.0327773
5: -6.8200684, -5.9123907, -6.8140545, -5.9181247, -0.5096638, 0.5046952
6: 8.7853680, 9.8561268, 8.7756586, 9.8289032, -0.5021470, 0.5255337
7: -13.7716475, -12.5765266, -13.7716112, -12.5787964, -0.5107598, 0.5134354
8: -5.8737149, -4.9862294, -5.8643279, -4.9753032, -0.4296453, 0.4160533
9: -10.3753424, -8.9364843, -10.3720741, -8.9299660, -1.0238891, 1.0133309

Time for backsubstitution: 8.73 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.39 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of NS_A1_B1_A1_B1_A1

### Relational analysis result of NS_A1_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3699926, upper bound: 0.3657920
time: 4.73 seconds

## Relational analysis of NS_A1_B1_A1_B1_A2

### Relational analysis result of NS_A1_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3730318, upper bound: 0.3724115
time: 3.46 seconds

## BFS NS instance: NS_A1_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.3032165, -0.1982946, -1.2977325, -0.1950707, -0.8269095, 0.8153749
1: -17.6943493, -16.2565098, -17.6869240, -16.2471333, -0.7513378, 0.7640598
2: -6.2651019, -4.9618049, -6.2656970, -4.9671540, -0.7910471, 0.7958021
3: -13.7244129, -12.5285845, -13.7167892, -12.5259399, -0.5277734, 0.5270994
4: -5.2241917, -4.0680456, -5.2451124, -4.0637407, -1.0502148, 1.0393982
5: -6.8210015, -5.9053082, -6.8098602, -5.9049082, -0.5074162, 0.5231363
6: 8.7853661, 9.8674965, 8.7795353, 9.8513050, -0.5078633, 0.5384400
7: -13.7719707, -12.5747986, -13.7722797, -12.5753918, -0.5148468, 0.5155735
8: -5.8806357, -4.9862289, -5.8781691, -4.9780350, -0.4418738, 0.4191089
9: -10.3824615, -8.9364853, -10.3857832, -8.9336576, -1.0301795, 1.0281806

Time for backsubstitution: 8.78 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of NS_A1_B1_A1_B2_A1

### Relational analysis result of NS_A1_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3803045, upper bound: 0.3758332
time: 3.54 seconds

## Relational analysis of NS_A1_B1_A1_B2_A2

### Relational analysis result of NS_A1_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3803045, upper bound: 0.3787837
time: 3.89 seconds

## BFS NS instance: NS_A1_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.2979411, -0.1965383, -1.2955616, -0.1893910, -0.8256617, 0.8165951
1: -17.6868038, -16.2531376, -17.6969604, -16.2608337, -0.7480381, 0.7588444
2: -6.2602167, -4.9672132, -6.2618113, -4.9668226, -0.7917738, 0.7922955
3: -13.7167625, -12.5368156, -13.7141113, -12.5432777, -0.5218837, 0.5236173
4: -5.2497768, -4.0643606, -5.2358012, -4.0680513, -1.0464783, 1.0487952
5: -6.8093185, -5.9154816, -6.8146086, -5.9184246, -0.5095079, 0.5090051
6: 8.7859993, 9.8438644, 8.7740211, 9.8289299, -0.5038253, 0.5378866
7: -13.7720442, -12.5763159, -13.7718639, -12.5786314, -0.5125186, 0.5136609
8: -5.8742104, -4.9748416, -5.8655238, -4.9694653, -0.4370469, 0.4222240
9: -10.3869028, -8.9335518, -10.3777828, -8.9296713, -1.0382390, 1.0147705

Time for backsubstitution: 8.98 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.43 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of NS_A1_B1_A2_B1_A1

### Relational analysis result of NS_A1_B1_A2_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3680981, upper bound: 0.3661522
time: 3.47 seconds

## Relational analysis of NS_A1_B1_A2_B1_A2

### Relational analysis result of NS_A1_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3710779, upper bound: 0.3730384
time: 3.35 seconds

## BFS NS instance: NS_A1_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.2996868, -0.1965373, -1.2980350, -0.1935654, -0.8273511, 0.8200722
1: -17.6868038, -16.2421532, -17.6869144, -16.2395630, -0.7577212, 0.7757733
2: -6.2617784, -4.9671412, -6.2648826, -4.9667225, -0.7938979, 0.7955503
3: -13.7167616, -12.5286055, -13.7170486, -12.5251532, -0.5280490, 0.5288787
4: -5.2624111, -4.0631814, -5.2630672, -4.0632229, -1.0611162, 1.0554194
5: -6.8102469, -5.9083991, -6.8104305, -5.9052086, -0.5072334, 0.5274670
6: 8.7859974, 9.8552151, 8.7779179, 9.8513298, -0.5095781, 0.5507383
7: -13.7723675, -12.5745888, -13.7725315, -12.5752296, -0.5165794, 0.5158045
8: -5.8811464, -4.9748392, -5.8793473, -4.9721956, -0.4491801, 0.4254525
9: -10.3940201, -8.9335527, -10.3914909, -8.9333649, -1.0444660, 1.0299058

Time for backsubstitution: 9.08 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.56 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of NS_A1_B1_A2_B2_A1

### Relational analysis result of NS_A1_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3784390, upper bound: 0.3765159
time: 3.54 seconds

## Relational analysis of NS_A1_B1_A2_B2_A2

### Relational analysis result of NS_A1_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3784390, upper bound: 0.3798819
time: 3.45 seconds

## BFS NS instance: NS_A1_B2_A1_B1

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

Time for backsubstitution: 9.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.48 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of NS_A1_B2_A1_B1_A1

### Relational analysis result of NS_A1_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3596636, upper bound: 0.3534278
time: 4.48 seconds

## Relational analysis of NS_A1_B2_A1_B1_A2

### Relational analysis result of NS_A1_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3673765, upper bound: 0.3659192
time: 3.84 seconds

## BFS NS instance: NS_A1_B2_A1_B2

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

Time for backsubstitution: 9.06 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of NS_A1_B2_A1_B2_A1

### Relational analysis result of NS_A1_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3741317, upper bound: 0.3696200
time: 3.73 seconds

## Relational analysis of NS_A1_B2_A1_B2_A2

### Relational analysis result of NS_A1_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3741317, upper bound: 0.3696200
time: 4.27 seconds

## BFS NS instance: NS_A1_B2_A2_B1

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

Time for backsubstitution: 9.33 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.58 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of NS_A1_B2_A2_B1_A1

### Relational analysis result of NS_A1_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3578787, upper bound: 0.3539889
time: 3.74 seconds

## Relational analysis of NS_A1_B2_A2_B1_A2

### Relational analysis result of NS_A1_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3659549, upper bound: 0.3673485
time: 4.30 seconds

## BFS NS instance: NS_A1_B2_A2_B2

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

Time for backsubstitution: 9.45 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.58 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of NS_A1_B2_A2_B2_A1

### Relational analysis result of NS_A1_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3728278, upper bound: 0.3711665
time: 4.64 seconds

## Relational analysis of NS_A1_B2_A2_B2_A2

### Relational analysis result of NS_A1_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3728278, upper bound: 0.3731797
time: 3.72 seconds

## BFS NS instance: NS_A2_B1_A1_B1

### Backsubstitution after applying NS history:
0: -1.2930579, -0.1848934, -1.2952582, -0.1908982, -0.8209805, 0.8291969
1: -17.6859131, -16.2610512, -17.6969719, -16.2683506, -0.7290738, 0.7515676
2: -6.3100328, -4.9891500, -6.2627058, -4.9672551, -0.8530564, 0.7910671
3: -13.7431602, -12.5398273, -13.7138548, -12.5440683, -0.5445819, 0.5217826
4: -5.1941981, -4.0430269, -5.2178564, -4.0685368, -1.0287800, 1.0665717
5: -6.8117008, -5.9096427, -6.8140545, -5.9181247, -0.5022810, 0.5072880
6: 8.8379097, 9.9218569, 8.7756586, 9.8289032, -0.4828374, 0.6276886
7: -13.7962799, -12.5856905, -13.7716112, -12.5787964, -0.5447936, 0.5128686
8: -5.8610072, -4.9687586, -5.8643279, -4.9753032, -0.4117184, 0.4252226
9: -10.3854504, -8.9528074, -10.3720741, -8.9299660, -1.0265985, 0.9909868

Time for backsubstitution: 9.45 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.58 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of NS_A2_B1_A1_B1_A1

### Relational analysis result of NS_A2_B1_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3586151, upper bound: 0.3557182
time: 3.56 seconds

## Relational analysis of NS_A2_B1_A1_B1_A2

### Relational analysis result of NS_A2_B1_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3669471, upper bound: 0.3668588
time: 6.12 seconds

## BFS NS instance: NS_A2_B1_A1_B2

### Backsubstitution after applying NS history:
0: -1.2947941, -0.1848926, -1.2977325, -0.1950707, -0.8226886, 0.8324823
1: -17.6859112, -16.2501564, -17.6869240, -16.2471333, -0.7403557, 0.7681692
2: -6.3115549, -4.9890747, -6.2656970, -4.9671540, -0.8551769, 0.7943292
3: -13.7431602, -12.5316162, -13.7167892, -12.5259399, -0.5507812, 0.5270426
4: -5.2073154, -4.0418744, -5.2451124, -4.0637407, -1.0434761, 1.0731626
5: -6.8125925, -5.9025588, -6.8098602, -5.9049082, -0.5002487, 0.5253162
6: 8.8379059, 9.9327297, 8.7795353, 9.8513050, -0.4898894, 0.6400278
7: -13.7966022, -12.5839319, -13.7722797, -12.5753918, -0.5488634, 0.5149283
8: -5.8678856, -4.9687586, -5.8781691, -4.9780350, -0.4239404, 0.4281076
9: -10.3927917, -8.9528065, -10.3857832, -8.9336576, -1.0330710, 1.0058422

Time for backsubstitution: 9.51 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.58 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of NS_A2_B1_A1_B2_A1

### Relational analysis result of NS_A2_B1_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3735641, upper bound: 0.3694069
time: 4.32 seconds

## Relational analysis of NS_A2_B1_A1_B2_A2

### Relational analysis result of NS_A2_B1_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3735641, upper bound: 0.3720171
time: 4.43 seconds

## BFS NS instance: NS_A2_B1_A2_B1

### Backsubstitution after applying NS history:
0: -1.2895542, -0.1831353, -1.2955616, -0.1893910, -0.8214831, 0.8337340
1: -17.6783695, -16.2467861, -17.6969604, -16.2608337, -0.7358630, 0.7634213
2: -6.3066750, -4.9945450, -6.2618113, -4.9668226, -0.8559103, 0.7907896
3: -13.7354860, -12.5398474, -13.7141113, -12.5432777, -0.5448813, 0.5235884
4: -5.2323246, -4.0381656, -5.2358012, -4.0680513, -1.0396881, 1.0825725
5: -6.8009233, -5.9127297, -6.8146086, -5.9184246, -0.5020988, 0.5115664
6: 8.8385820, 9.9095926, 8.7740211, 9.8289299, -0.4842402, 0.6400433
7: -13.7966757, -12.5854855, -13.7718639, -12.5786314, -0.5465486, 0.5130193
8: -5.8614006, -4.9573712, -5.8655238, -4.9694653, -0.4190580, 0.4313922
9: -10.3969393, -8.9499102, -10.3777828, -8.9296713, -1.0409522, 0.9924226

Time for backsubstitution: 9.46 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.56 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of NS_A2_B1_A2_B1_A1

### Relational analysis result of NS_A2_B1_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3568394, upper bound: 0.3562826
time: 3.37 seconds

## Relational analysis of NS_A2_B1_A2_B1_A2

### Relational analysis result of NS_A2_B1_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3660217, upper bound: 0.3678959
time: 3.91 seconds

## BFS NS instance: NS_A2_B1_A2_B2

### Backsubstitution after applying NS history:
0: -1.2912927, -0.1831342, -1.2980350, -0.1935654, -0.8231826, 0.8371811
1: -17.6783695, -16.2357941, -17.6869144, -16.2395630, -0.7467422, 0.7799232
2: -6.3082333, -4.9944720, -6.2648826, -4.9667225, -0.8580277, 0.7940440
3: -13.7354870, -12.5316391, -13.7170486, -12.5251532, -0.5510550, 0.5288482
4: -5.2454529, -4.0369992, -5.2630672, -4.0632229, -1.0543337, 1.0891695
5: -6.8018141, -5.9056463, -6.8104305, -5.9052086, -0.5000463, 0.5296150
6: 8.8385811, 9.9204502, 8.7779179, 9.8513298, -0.4913288, 0.6523290
7: -13.7969971, -12.5837250, -13.7725315, -12.5752296, -0.5505927, 0.5150847
8: -5.8682728, -4.9573698, -5.8793473, -4.9721956, -0.4311829, 0.4344509
9: -10.4042788, -8.9499111, -10.3914909, -8.9333649, -1.0473633, 1.0075631

Time for backsubstitution: 9.31 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.57 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of NS_A2_B1_A2_B2_A1

### Relational analysis result of NS_A2_B1_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3725761, upper bound: 0.3707665
time: 5.39 seconds

## Relational analysis of NS_A2_B1_A2_B2_A2

### Relational analysis result of NS_A2_B1_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3725761, upper bound: 0.3733885
time: 4.25 seconds

## BFS NS instance: NS_A2_B2_A1_B1

### Backsubstitution after applying NS history:
0: -1.2930579, -0.1848934, -1.2868850, -0.1775031, -0.8257933, 0.8126884
1: -17.6859131, -16.2610512, -17.6885319, -16.2620106, -0.7402313, 0.7460237
2: -6.3100328, -4.9891500, -6.3091636, -4.9945917, -0.7939234, 0.7975311
3: -13.7431602, -12.5398273, -13.7325850, -12.5470982, -0.5247226, 0.5249641
4: -5.1941981, -4.0430269, -5.1999159, -4.0433788, -1.0325270, 1.0297637
5: -6.8117008, -5.9096427, -6.8057051, -5.9153771, -0.5109296, 0.5059752
6: 8.8379097, 9.9218569, 8.8268661, 9.8951445, -0.5080798, 0.5314667
7: -13.7962799, -12.5856905, -13.7962418, -12.5880156, -0.5139017, 0.5165887
8: -5.8610072, -4.9687586, -5.8516335, -4.9578314, -0.4317858, 0.4181824
9: -10.3854504, -8.9528074, -10.3817873, -8.9463177, -1.0216560, 1.0111356

Time for backsubstitution: 9.14 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of NS_A2_B2_A1_B1_A1

### Relational analysis result of NS_A2_B2_A1_B1_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3586151, upper bound: 0.3516782
time: 3.26 seconds

## Relational analysis of NS_A2_B2_A1_B1_A2

### Relational analysis result of NS_A2_B2_A1_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3669471, upper bound: 0.3636589
time: 3.82 seconds

## BFS NS instance: NS_A2_B2_A1_B2

### Backsubstitution after applying NS history:
0: -1.2947941, -0.1848926, -1.2893398, -0.1816746, -0.8275232, 0.8159800
1: -17.6859112, -16.2501564, -17.6784878, -16.2407837, -0.7503152, 0.7630403
2: -6.3115549, -4.9890747, -6.3121486, -4.9944887, -0.7960641, 0.8008161
3: -13.7431602, -12.5316162, -13.7355118, -12.5289745, -0.5309205, 0.5302167
4: -5.2073154, -4.0418744, -5.2284575, -4.0375586, -1.0472164, 1.0364032
5: -6.8125925, -5.9025588, -6.8014431, -5.9021606, -0.5085983, 0.5245062
6: 8.8379059, 9.9327297, 8.8320818, 9.9162483, -0.5137864, 0.5443780
7: -13.7966022, -12.5839319, -13.7969084, -12.5845394, -0.5180001, 0.5187280
8: -5.8678856, -4.9687586, -5.8653331, -4.9605637, -0.4440143, 0.4212459
9: -10.3927917, -8.9528065, -10.3959808, -8.9500103, -1.0279484, 1.0259705

Time for backsubstitution: 9.12 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.46 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of NS_A2_B2_A1_B2_A1

### Relational analysis result of NS_A2_B2_A1_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3735642, upper bound: 0.3668241
time: 4.37 seconds

## Relational analysis of NS_A2_B2_A1_B2_A2

### Relational analysis result of NS_A2_B2_A1_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3735642, upper bound: 0.3691173
time: 3.63 seconds

## BFS NS instance: NS_A2_B2_A2_B1

### Backsubstitution after applying NS history:
0: -1.2895542, -0.1831353, -1.2871927, -0.1760002, -0.8262277, 0.8172216
1: -17.6783695, -16.2467861, -17.6885262, -16.2545013, -0.7469783, 0.7578449
2: -6.3066750, -4.9945450, -6.3082738, -4.9941683, -0.7967818, 0.7972703
3: -13.7354860, -12.5398474, -13.7328405, -12.5463076, -0.5250027, 0.5267603
4: -5.2323246, -4.0381656, -5.2178097, -4.0428829, -1.0434632, 1.0457740
5: -6.8009233, -5.9127297, -6.8062449, -5.9156694, -0.5108011, 0.5102696
6: 8.8385820, 9.9095926, 8.8252869, 9.8951683, -0.5097628, 0.5438342
7: -13.7966757, -12.5854855, -13.7964935, -12.5878563, -0.5156589, 0.5168140
8: -5.8614006, -4.9573712, -5.8527746, -4.9519939, -0.4391832, 0.4243615
9: -10.3969393, -8.9499102, -10.3874626, -8.9460411, -1.0360336, 1.0125570

Time for backsubstitution: 9.15 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.48 seconds

### Candidate
type: A, layer: 3, pos: 2480

## Relational analysis of NS_A2_B2_A2_B1_A1

### Relational analysis result of NS_A2_B2_A2_B1_A1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3568393, upper bound: 0.3522391
time: 3.20 seconds

## Relational analysis of NS_A2_B2_A2_B1_A2

### Relational analysis result of NS_A2_B2_A2_B1_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3660217, upper bound: 0.3648596
time: 4.37 seconds

## BFS NS instance: NS_A2_B2_A2_B2

### Backsubstitution after applying NS history:
0: -1.2912927, -0.1831342, -1.2896476, -0.1801721, -0.8279424, 0.8206720
1: -17.6783695, -16.2357941, -17.6784782, -16.2332172, -0.7566586, 0.7747614
2: -6.3082333, -4.9944720, -6.3113394, -4.9940672, -0.7989192, 0.8005471
3: -13.7354870, -12.5316391, -13.7357740, -12.5281839, -0.5311742, 0.5320132
4: -5.2454529, -4.0369992, -5.2463632, -4.0370350, -1.0581036, 1.0524182
5: -6.8018141, -5.9056463, -6.8020010, -5.9024558, -0.5084407, 0.5288081
6: 8.8385811, 9.9204502, 8.8305254, 9.9162760, -0.5155065, 0.5566912
7: -13.7969971, -12.5837250, -13.7971601, -12.5843782, -0.5197310, 0.5189590
8: -5.8682728, -4.9573698, -5.8664589, -4.9547257, -0.4513140, 0.4275953
9: -10.4042788, -8.9499111, -10.4016552, -8.9497347, -1.0422621, 1.0276875

Time for backsubstitution: 9.11 seconds

### NS candidates at layer 1

No NS candidates found

### NS candidates at layer 3
type: A, layer: 3, pos: 1479
type: A, layer: 3, pos: 2480
type: A, layer: 3, pos: 1705
type: A, layer: 3, pos: 1502
type: A, layer: 3, pos: 2818
type: A, layer: 3, pos: 555
type: A, layer: 3, pos: 2901
type: A, layer: 3, pos: 76
type: A, layer: 3, pos: 1979
type: A, layer: 3, pos: 1775
type: A, layer: 3, pos: 3102
type: A, layer: 3, pos: 613
type: A, layer: 3, pos: 890
type: A, layer: 3, pos: 780
type: A, layer: 3, pos: 660
type: A, layer: 3, pos: 2858
type: A, layer: 3, pos: 1228
type: A, layer: 3, pos: 1515
type: A, layer: 3, pos: 1847
type: A, layer: 3, pos: 2396
type: A, layer: 3, pos: 38
type: A, layer: 3, pos: 1506
type: A, layer: 3, pos: 75
type: A, layer: 3, pos: 738
type: A, layer: 3, pos: 2827
type: A, layer: 3, pos: 192
type: A, layer: 3, pos: 330
type: A, layer: 3, pos: 2356
type: A, layer: 3, pos: 151
type: A, layer: 3, pos: 1406

Time for candidate selection: 0.42 seconds

### Candidate
type: A, layer: 3, pos: 1479

## Relational analysis of NS_A2_B2_A2_B2_A1

### Relational analysis result of NS_A2_B2_A2_B2_A1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3725761, upper bound: 0.3682849
time: 5.51 seconds

## Relational analysis of NS_A2_B2_A2_B2_A2

### Relational analysis result of NS_A2_B2_A2_B2_A2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3725761, upper bound: 0.3705886
time: 3.96 seconds

## Summary of splitting at layer (split count: 4)
- Time for NS candidates: 19.02 seconds
NS_A1_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3699926, upper bound: 0.3657920
NS_A1_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3730318, upper bound: 0.3724115
NS_A1_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3803045, upper bound: 0.3758332
NS_A1_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3803045, upper bound: 0.3787837
NS_A1_B1_A2_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3680981, upper bound: 0.3661522
NS_A1_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3710779, upper bound: 0.3730384
NS_A1_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3784390, upper bound: 0.3765159
NS_A1_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3784390, upper bound: 0.3798819
NS_A1_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3596636, upper bound: 0.3534278
NS_A1_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3673765, upper bound: 0.3659192
NS_A1_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3741317, upper bound: 0.3696200
NS_A1_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3741317, upper bound: 0.3696200
NS_A1_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3578787, upper bound: 0.3539889
NS_A1_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3659549, upper bound: 0.3673485
NS_A1_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3728278, upper bound: 0.3711665
NS_A1_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3728278, upper bound: 0.3731797
NS_A2_B1_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3586151, upper bound: 0.3557182
NS_A2_B1_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3669471, upper bound: 0.3668588
NS_A2_B1_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3735641, upper bound: 0.3694069
NS_A2_B1_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3735641, upper bound: 0.3720171
NS_A2_B1_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3568394, upper bound: 0.3562826
NS_A2_B1_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3660217, upper bound: 0.3678959
NS_A2_B1_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3725761, upper bound: 0.3707665
NS_A2_B1_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3725761, upper bound: 0.3733885
NS_A2_B2_A1_B1_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3586151, upper bound: 0.3516782
NS_A2_B2_A1_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3669471, upper bound: 0.3636589
NS_A2_B2_A1_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3735642, upper bound: 0.3668241
NS_A2_B2_A1_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3735642, upper bound: 0.3691173
NS_A2_B2_A2_B1_A1, status: Status.VERIFIED, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3568393, upper bound: 0.3522391
NS_A2_B2_A2_B1_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3660217, upper bound: 0.3648596
NS_A2_B2_A2_B2_A1, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3725761, upper bound: 0.3682849
NS_A2_B2_A2_B2_A2, status: Status.UNKNOWN, split count: 5, time: 19.02
Output dim: 6, lower bound: -0.3725761, upper bound: 0.3705886

## NS Result
status: Status.UNKNOWN
execution time: (base) + (ns) = 57.91 + 546.39 = 604.30 seconds
