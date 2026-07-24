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
execution time: IAR + RelationalAnalysis = 24.30 + 35.96 = 60.26 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3983825, upper bound: 0.3983820

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 1502

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1228

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3981751, upper bound: 0.3982186
time: 3.35 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3982172, upper bound: 0.3981765
time: 3.21 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 6.58 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 6.58
Output dim: 6, lower bound: -0.3981751, upper bound: 0.3982186
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 6.58
Output dim: 6, lower bound: -0.3982172, upper bound: 0.3981765

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8351326, 0.8351965
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7916994, 0.7915769
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8051097, 0.8056359
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5413718, 0.5417798
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0939274, 1.0934916
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5340154, 0.5335108
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5540631, 0.5540795
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5211735, 0.5212381
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4542551, 0.4541036
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0434418, 1.0434279

Time for backsubstitution: 9.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1775

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 38

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3966026, upper bound: 0.3982059
time: 3.31 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3981624, upper bound: 0.3966461
time: 3.35 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8351960, 0.8351331
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7915769, 0.7916994
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8056362, 0.8051095
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5417798, 0.5413716
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0934916, 1.0939279
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5335107, 0.5340154
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5540795, 0.5540628
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5212383, 0.5211737
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4541035, 0.4542551
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0434284, 1.0434413

Time for backsubstitution: 8.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 555

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 75

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3740018, upper bound: 0.3739608
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3740018, upper bound: 0.3739608
time: 3.61 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 15.79 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.79
Output dim: 6, lower bound: -0.3966026, upper bound: 0.3982059
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.79
Output dim: 6, lower bound: -0.3981624, upper bound: 0.3966461
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 15.79
Output dim: 6, lower bound: -0.3740018, upper bound: 0.3739608
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 15.79
Output dim: 6, lower bound: -0.3740018, upper bound: 0.3739608

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8366160, 0.8369517
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7914293, 0.7912602
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8048842, 0.8054657
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5403030, 0.5419307
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0925665, 1.0920768
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5346029, 0.5341488
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5530987, 0.5531917
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5160580, 0.5166402
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4531068, 0.4533832
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0425091, 1.0425172

Time for backsubstitution: 9.46 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1515

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 330

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3895357, upper bound: 0.3911229
time: 3.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3895357, upper bound: 0.3911229
time: 3.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8368878, 0.8366795
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7913826, 0.7913070
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8049390, 0.8054104
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5415227, 0.5407112
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0925131, 1.0921307
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5346534, 0.5340984
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5531750, 0.5531149
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5165758, 0.5161223
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4535347, 0.4529552
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0425310, 1.0424957

Time for backsubstitution: 9.40 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 2356

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 151

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3979285, upper bound: 0.3939556
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3955052, upper bound: 0.3964283
time: 5.64 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8329263, 0.8350625
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7912436, 0.7921977
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8055141, 0.8062019
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5417619, 0.5416107
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0934372, 1.0943990
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5333116, 0.5339704
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5544879, 0.5540547
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5211172, 0.5231726
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4549247, 0.4541483
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0434093, 1.0439949

Time for backsubstitution: 8.72 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 330

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3662332, upper bound: 0.3661904
time: 3.01 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3662332, upper bound: 0.3661904
time: 2.88 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8351259, 0.8351331
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7915769, 0.7913661
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8056362, 0.8049874
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5417798, 0.5413537
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0934916, 1.0938735
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5334656, 0.5340154
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5540712, 0.5540628
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5212383, 0.5210526
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4539968, 0.4542551
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0434284, 1.0434232

Time for backsubstitution: 8.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 3102

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3711041, upper bound: 0.3734404
time: 3.55 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3734813, upper bound: 0.3710611
time: 3.09 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 15.31 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.31
Output dim: 6, lower bound: -0.3895357, upper bound: 0.3911229
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.31
Output dim: 6, lower bound: -0.3895357, upper bound: 0.3911229
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.31
Output dim: 6, lower bound: -0.3979285, upper bound: 0.3939556
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.31
Output dim: 6, lower bound: -0.3955052, upper bound: 0.3964283
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.31
Output dim: 6, lower bound: -0.3662332, upper bound: 0.3661904
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.31
Output dim: 6, lower bound: -0.3662332, upper bound: 0.3661904
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 15.31
Output dim: 6, lower bound: -0.3711041, upper bound: 0.3734404
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 15.31
Output dim: 6, lower bound: -0.3734813, upper bound: 0.3710611

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8365960, 0.8370390
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7914116, 0.7912407
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8046682, 0.8054380
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5403290, 0.5419216
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0925636, 1.0920615
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5347319, 0.5341291
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5530963, 0.5531926
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5158212, 0.5166125
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4531010, 0.4534595
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0424995, 1.0424175

Time for backsubstitution: 9.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 780

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 660

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3844155, upper bound: 0.3908882
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3893015, upper bound: 0.3859921
time: 3.83 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8366160, 0.8369312
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7914102, 0.7912602
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8048565, 0.8054657
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5402942, 0.5419307
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0925665, 1.0920734
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5345833, 0.5341488
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5530987, 0.5531893
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5160303, 0.5166402
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4531068, 0.4533776
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0425091, 1.0425076

Time for backsubstitution: 8.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 2901

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3894987, upper bound: 0.3899671
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3883697, upper bound: 0.3910858
time: 3.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8368864, 0.8366776
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7913823, 0.7913060
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8049395, 0.8054113
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5415225, 0.5407109
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0925131, 1.0921302
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5346532, 0.5340978
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5531759, 0.5531149
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5165758, 0.5161226
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4535347, 0.4529550
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0425310, 1.0424967

Time for backsubstitution: 9.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 332

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1406

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3979285, upper bound: 0.3936265
time: 3.30 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3975905, upper bound: 0.3939556
time: 3.42 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8368864, 0.8366780
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7913814, 0.7913065
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8049400, 0.8054109
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5415225, 0.5407109
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0925121, 1.0921302
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5346529, 0.5340979
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5531750, 0.5531158
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5165758, 0.5161226
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4535347, 0.4529551
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0425310, 1.0424962

Time for backsubstitution: 8.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2130

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 76

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3934970, upper bound: 0.3939007
time: 3.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3929810, upper bound: 0.3944168
time: 3.42 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8329062, 0.8351502
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7912261, 0.7921784
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8052979, 0.8061738
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5417879, 0.5416019
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0934334, 1.0943832
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5334396, 0.5339499
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5544863, 0.5540564
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5208802, 0.5231447
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4549195, 0.4542248
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0433998, 1.0438943

Time for backsubstitution: 8.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1705

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 151

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3658666, upper bound: 0.3635298
time: 2.87 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3635367, upper bound: 0.3658238
time: 2.85 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8329263, 0.8350425
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7912247, 0.7921977
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8054862, 0.8062019
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5417531, 0.5416107
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0934372, 1.0943956
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5332911, 0.5339704
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5544879, 0.5540531
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5210893, 0.5231726
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4549247, 0.4541428
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0434093, 1.0439844

Time for backsubstitution: 8.85 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1506

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2396

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3620119, upper bound: 0.3608200
time: 2.92 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3608626, upper bound: 0.3619688
time: 3.94 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8361397, 0.8363271
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7953107, 0.7947788
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8032951, 0.8024158
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5405076, 0.5403280
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0879869, 1.0881643
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5324948, 0.5327389
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5523868, 0.5525303
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5154786, 0.5137188
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4540408, 0.4543765
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0442724, 1.0444651

Time for backsubstitution: 8.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 1979

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 890

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3708596, upper bound: 0.3733251
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3708585, upper bound: 0.3727377
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8363204, 0.8361464
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7949893, 0.7951002
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8030634, 0.8026476
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5407543, 0.5400815
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0877838, 1.0883684
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5321887, 0.5330448
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5525386, 0.5523784
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5139041, 0.5152931
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4541180, 0.4542993
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0444698, 1.0442681

Time for backsubstitution: 9.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 890

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3727789, upper bound: 0.3708155
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3733676, upper bound: 0.3708165
time: 2.98 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 15.20 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3844155, upper bound: 0.3908882
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3893015, upper bound: 0.3859921
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3894987, upper bound: 0.3899671
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3883697, upper bound: 0.3910858
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3979285, upper bound: 0.3936265
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3975905, upper bound: 0.3939556
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3934970, upper bound: 0.3939007
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3929810, upper bound: 0.3944168
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3658666, upper bound: 0.3635298
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3635367, upper bound: 0.3658238
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3620119, upper bound: 0.3608200
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3608626, upper bound: 0.3619688
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3708596, upper bound: 0.3733251
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3708585, upper bound: 0.3727377
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3727789, upper bound: 0.3708155
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 15.20
Output dim: 6, lower bound: -0.3733676, upper bound: 0.3708165

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8370585, 0.8330169
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7624304, 0.7583380
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8030243, 0.8025966
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5236237, 0.5239496
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0897617, 1.0907636
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5040810, 0.5079374
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5453713, 0.5484734
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5143256, 0.5147343
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4456640, 0.4471734
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0205617, 1.0191936

Time for backsubstitution: 8.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 613

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2356

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3819638, upper bound: 0.3897949
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3833409, upper bound: 0.3883894
time: 3.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8325739, 0.8375020
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7585089, 0.7622597
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8018270, 0.8037930
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5223567, 0.5252162
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0912647, 1.0892596
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5085399, 0.5034785
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5483770, 0.5454674
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5139430, 0.5151172
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4468149, 0.4460224
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0192761, 1.0204797

Time for backsubstitution: 9.33 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 3102

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1515

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3864702, upper bound: 0.3859294
time: 3.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3892374, upper bound: 0.3822381
time: 3.73 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8339829, 0.8343840
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7904723, 0.7901978
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7968686, 0.7978392
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5333591, 0.5352247
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0925541, 1.0921283
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5305316, 0.5293084
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5530472, 0.5531259
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5143287, 0.5139344
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4474280, 0.4489965
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0422745, 1.0422883

Time for backsubstitution: 8.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 2818

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 192

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3717975, upper bound: 0.3713479
time: 5.24 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3717975, upper bound: 0.3713479
time: 5.20 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8340688, 0.8342977
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7903478, 0.7903223
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7972295, 0.7974777
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5335886, 0.5349951
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0926218, 1.0920610
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5297430, 0.5300972
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5530353, 0.5531378
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5133243, 0.5149388
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4487257, 0.4476988
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0422907, 1.0422726

Time for backsubstitution: 8.71 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 1406

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 660

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3832581, upper bound: 0.3908512
time: 3.84 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3881364, upper bound: 0.3859574
time: 4.19 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8368850, 0.8366752
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7913785, 0.7913032
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8049400, 0.8054132
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5415142, 0.5406971
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0924902, 1.0921130
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5346754, 0.5341088
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5531769, 0.5531154
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5165548, 0.5161085
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4535096, 0.4529338
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0425286, 1.0424962

Time for backsubstitution: 8.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1705

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3930883, upper bound: 0.3890626
time: 3.28 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3933756, upper bound: 0.3888172
time: 3.38 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8368840, 0.8366761
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7913795, 0.7913024
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8049409, 0.8054123
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5415087, 0.5407026
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0924959, 1.0921078
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5346642, 0.5341198
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5531764, 0.5531163
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5165617, 0.5161014
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4535134, 0.4529300
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0425305, 1.0424943

Time for backsubstitution: 8.68 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 738

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 660

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3925970, upper bound: 0.3937198
time: 3.22 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3973527, upper bound: 0.3889926
time: 3.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8363476, 0.8360190
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7912722, 0.7912185
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8048973, 0.8054008
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5413189, 0.5400348
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0919037, 1.0918241
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5346489, 0.5340741
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5528746, 0.5516133
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5165029, 0.5159621
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4535156, 0.4530421
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0420923, 1.0414877

Time for backsubstitution: 8.66 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2818

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 192

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751220, upper bound: 0.3763030
time: 3.49 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3751220, upper bound: 0.3763030
time: 3.46 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8368864, 0.8361392
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7912941, 0.7913065
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8049297, 0.8054109
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5415225, 0.5405073
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0922070, 1.0921302
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5346529, 0.5340939
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5516725, 0.5531158
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5165758, 0.5160496
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4535347, 0.4529359
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0425310, 1.0420570

Time for backsubstitution: 8.59 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1705

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3881413, upper bound: 0.3898931
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3884324, upper bound: 0.3896325
time: 3.50 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8329053, 0.8351488
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7912252, 0.7921770
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8052979, 0.8061748
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5417879, 0.5416021
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0934334, 1.0943823
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5334394, 0.5339495
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5544872, 0.5540564
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5208805, 0.5231447
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4549192, 0.4542247
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0433998, 1.0438952

Time for backsubstitution: 8.60 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 38

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 890

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3651650, upper bound: 0.3633942
time: 2.99 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3657308, upper bound: 0.3628277
time: 3.04 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8329048, 0.8351493
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7912247, 0.7921774
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8052988, 0.8061743
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5417879, 0.5416021
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0934334, 1.0943823
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5334392, 0.5339497
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5544865, 0.5540574
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5208805, 0.5231447
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4549192, 0.4542248
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0433998, 1.0438948

Time for backsubstitution: 8.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 332

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1515

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3599344, upper bound: 0.3657599
time: 2.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3634726, upper bound: 0.3622218
time: 3.02 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8327217, 0.8346128
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7901745, 0.7915070
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8048153, 0.8054895
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5413730, 0.5410168
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0933180, 1.0943880
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5326455, 0.5333867
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5543509, 0.5539069
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5209689, 0.5231013
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4548997, 0.4541073
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0433316, 1.0439792

Time for backsubstitution: 9.19 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1775

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 890

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3613124, upper bound: 0.3606860
time: 3.68 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3618782, upper bound: 0.3601220
time: 3.30 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8324971, 0.8348379
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7905340, 0.7911472
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8047738, 0.8055305
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5411592, 0.5412307
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0934296, 1.0942769
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5327079, 0.5333242
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5543423, 0.5539160
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5210180, 0.5230525
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4548894, 0.4541175
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0434051, 1.0439067

Time for backsubstitution: 8.63 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1479

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 738

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3571208, upper bound: 0.3615098
time: 4.32 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3604036, upper bound: 0.3604064
time: 3.84 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8356667, 0.8358006
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7951233, 0.7944496
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8030243, 0.8021541
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5401030, 0.5399365
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0869226, 1.0877838
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5321219, 0.5327886
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5508902, 0.5510201
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5083385, 0.5050600
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4539325, 0.4542505
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0397720, 1.0395179

Time for backsubstitution: 9.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1406

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2396

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3668346, upper bound: 0.3679612
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3655607, upper bound: 0.3691131
time: 3.36 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8356133, 0.8358541
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7949669, 0.7945912
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8030348, 0.8021441
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5401161, 0.5399234
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0872355, 1.0871000
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5322652, 0.5323658
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5508759, 0.5510340
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5068202, 0.5065029
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4539149, 0.4542680
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0393276, 1.0397210

Time for backsubstitution: 8.61 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2858

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3552733, upper bound: 0.3554959
time: 3.10 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 6, lower bound: -0.3552733, upper bound: 0.3554959
time: 2.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8358474, 0.8356204
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7948020, 0.7947557
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8027930, 0.8023858
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5403497, 0.5396898
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0867186, 1.0876164
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5318160, 0.5328149
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5510421, 0.5508680
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5066881, 0.5066345
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4540098, 0.4541732
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0397263, 1.0393209

Time for backsubstitution: 8.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 660

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 151

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3724169, upper bound: 0.3681460
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3700558, upper bound: 0.3705193
time: 4.19 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8357940, 0.8356738
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7946608, 0.7949126
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8028030, 0.8023753
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5403626, 0.5396767
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0874023, 1.0873036
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5322390, 0.5326717
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5510280, 0.5508823
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5052457, 0.5081527
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4539921, 0.4541909
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0395241, 1.0397663

Time for backsubstitution: 9.27 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 660

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 330

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3656511, upper bound: 0.3630619
time: 3.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3656511, upper bound: 0.3630619
time: 3.02 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 15.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3819638, upper bound: 0.3897949
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3833409, upper bound: 0.3883894
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3864702, upper bound: 0.3859294
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3892374, upper bound: 0.3822381
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3717975, upper bound: 0.3713479
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3717975, upper bound: 0.3713479
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3832581, upper bound: 0.3908512
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3881364, upper bound: 0.3859574
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3930883, upper bound: 0.3890626
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3933756, upper bound: 0.3888172
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3925970, upper bound: 0.3937198
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3973527, upper bound: 0.3889926
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3751220, upper bound: 0.3763030
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3751220, upper bound: 0.3763030
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3881413, upper bound: 0.3898931
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3884324, upper bound: 0.3896325
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3651650, upper bound: 0.3633942
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3657308, upper bound: 0.3628277
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3599344, upper bound: 0.3657599
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3634726, upper bound: 0.3622218
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3613124, upper bound: 0.3606860
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3618782, upper bound: 0.3601220
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3571208, upper bound: 0.3615098
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3604036, upper bound: 0.3604064
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3668346, upper bound: 0.3679612
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3655607, upper bound: 0.3691131
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.VERIFIED, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3552733, upper bound: 0.3554959
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3552733, upper bound: 0.3554959
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3724169, upper bound: 0.3681460
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3700558, upper bound: 0.3705193
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3656511, upper bound: 0.3630619
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 15.30
Output dim: 6, lower bound: -0.3656511, upper bound: 0.3630619

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8370566, 0.8330164
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7624290, 0.7583389
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8030269, 0.8025970
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5236247, 0.5239496
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0897589, 1.0907626
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5040808, 0.5079376
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5453711, 0.5484786
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5143268, 0.5147328
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4456614, 0.4471722
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0205626, 1.0191927

Time for backsubstitution: 8.62 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 1502

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 2818

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3792806, upper bound: 0.3871870
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3793562, upper bound: 0.3871089
time: 3.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8370581, 0.8330169
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7624304, 0.7583368
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8030245, 0.8025966
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5236237, 0.5239496
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0897608, 1.0907636
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5040812, 0.5079374
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5453713, 0.5484731
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5143242, 0.5147343
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4456629, 0.4471734
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0205607, 1.0191936

Time for backsubstitution: 8.65 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2858

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 151

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Optimization infeasible because this subproblem isn't reachable.

### Candidate
type: DSZ, layer: 3, pos: 1479

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3792630, upper bound: 0.3826087
time: 3.52 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3778862, upper bound: 0.3844267
time: 3.50 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8084149, 0.8185177
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7556217, 0.7584338
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8104887, 0.8143268
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5247161, 0.5281190
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0848789, 1.0795240
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5119884, 0.5057261
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5434496, 0.5418158
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5117126, 0.5128827
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4464335, 0.4457350
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0132732, 1.0144067

Time for backsubstitution: 8.67 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 2480

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 3, pos: 1705

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3818410, upper bound: 0.3813837
time: 4.06 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3820373, upper bound: 0.3810913
time: 3.53 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8135600, 0.8133430
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7546828, 0.7593727
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.8123603, 0.8123636
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5252597, 0.5275754
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0815296, 1.0828738
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5107377, 0.5069269
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5447252, 0.5405400
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.5117087, 0.5128865
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4465275, 0.4456410
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0132027, 1.0144629

Time for backsubstitution: 8.62 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 60.26 + 544.65 = 604.91 seconds
