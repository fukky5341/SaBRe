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
execution time: IAR + RelationalAnalysis = 22.98 + 35.89 = 58.87 seconds
status: Status.UNKNOWN
relational distance
Output dim: 6, lower bound: -0.3983825, upper bound: 0.3983820

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 2480
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.41 seconds

### Candidate
type: DSZ, layer: 3, pos: 2480

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3937107, upper bound: 0.3918690
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3918677, upper bound: 0.3937121
time: 3.49 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.53 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.53
Output dim: 6, lower bound: -0.3937107, upper bound: 0.3918690
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.53
Output dim: 6, lower bound: -0.3918677, upper bound: 0.3937121

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8349695, 0.8349524
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7224100, 0.7297354
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7569776, 0.7503624
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5034747, 0.5095108
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0601320, 1.0588121
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5276558, 0.5276084
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5431004, 0.5405579
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3918438, 0.3859761
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4374758, 0.4372705
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0558996, 1.0551314

Time for backsubstitution: 8.22 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 1502

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3840898, upper bound: 0.3822943
time: 3.18 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3840898, upper bound: 0.3822943
time: 3.29 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8349524, 0.8349695
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7297356, 0.7224102
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7503619, 0.7569771
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5095108, 0.5034746
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0588121, 1.0601315
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5276086, 0.5276558
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5405579, 0.5431004
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3859762, 0.3918437
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4372708, 0.4374758
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0551310, 1.0558996

Time for backsubstitution: 8.86 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1502
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 1502

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3822945, upper bound: 0.3840911
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3822945, upper bound: 0.3840911
time: 3.51 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 16.03 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.03
Output dim: 6, lower bound: -0.3840898, upper bound: 0.3822943
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.03
Output dim: 6, lower bound: -0.3840898, upper bound: 0.3822943
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 16.03
Output dim: 6, lower bound: -0.3822945, upper bound: 0.3840911
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 16.03
Output dim: 6, lower bound: -0.3822945, upper bound: 0.3840911

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8348613, 0.8346324
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7216458, 0.7281835
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7554131, 0.7488999
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5023947, 0.5083083
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0591831, 1.0584278
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5280159, 0.5258435
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5375081, 0.5366774
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3899255, 0.3831954
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4326788, 0.4383121
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0556593, 1.0511003

Time for backsubstitution: 8.70 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 3, pos: 332

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3789472, upper bound: 0.3778531
time: 3.77 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3789816, upper bound: 0.3776212
time: 3.28 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8349695, 0.8348441
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7208581, 0.7297354
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7569776, 0.7487979
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5034747, 0.5084307
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0601320, 1.0578637
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5258908, 0.5276084
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5392195, 0.5405579
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3918438, 0.3840578
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4374758, 0.4324735
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0518694, 1.0551314

Time for backsubstitution: 9.01 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 332

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3789472, upper bound: 0.3778531
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3789816, upper bound: 0.3776212
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8348441, 0.8346496
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7289720, 0.7208581
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7487979, 0.7555175
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5084307, 0.5022722
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0578642, 1.0597472
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5279684, 0.5258908
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5349666, 0.5392194
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3840579, 0.3890632
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4324735, 0.4385173
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0548906, 1.0518689

Time for backsubstitution: 9.02 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 332

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3776199, upper bound: 0.3789815
time: 3.77 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3778533, upper bound: 0.3789486
time: 3.79 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8349524, 0.8348613
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.7281833, 0.7224102
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7503619, 0.7554126
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5095108, 0.5023946
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0588121, 1.0591831
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5258434, 0.5276558
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5366775, 0.5431004
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3859762, 0.3899254
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4372708, 0.4326787
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0511007, 1.0558996

Time for backsubstitution: 9.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 332
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 332

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3776199, upper bound: 0.3789815
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3778533, upper bound: 0.3789486
time: 3.42 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 16.43 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.43
Output dim: 6, lower bound: -0.3789472, upper bound: 0.3778531
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.43
Output dim: 6, lower bound: -0.3789816, upper bound: 0.3776212
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.43
Output dim: 6, lower bound: -0.3789472, upper bound: 0.3778531
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.43
Output dim: 6, lower bound: -0.3789816, upper bound: 0.3776212
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.43
Output dim: 6, lower bound: -0.3776199, upper bound: 0.3789815
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.43
Output dim: 6, lower bound: -0.3778533, upper bound: 0.3789486
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 16.43
Output dim: 6, lower bound: -0.3776199, upper bound: 0.3789815
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 16.43
Output dim: 6, lower bound: -0.3778533, upper bound: 0.3789486

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8289928, 0.8301220
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6849251, 0.6872153
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7546954, 0.7479534
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4927483, 0.4996841
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0244207, 1.0360942
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5167184, 0.5163381
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5244615, 0.5221789
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3903277, 0.3838153
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4203067, 0.4231576
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0487309, 1.0497046

Time for backsubstitution: 9.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 660

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3739477, upper bound: 0.3776179
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3787069, upper bound: 0.3730398
time: 3.43 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8303509, 0.8287702
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6806779, 0.6880274
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7544317, 0.7481823
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4937704, 0.4988908
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0370636, 1.0236654
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5185103, 0.5141957
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5236428, 0.5236304
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3904092, 0.3835976
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4175243, 0.4263787
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0537519, 1.0441732

Time for backsubstitution: 9.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 660

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3739839, upper bound: 0.3773879
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3787432, upper bound: 0.3728272
time: 3.52 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8290739, 0.8303337
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6844809, 0.6890469
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7562594, 0.7478161
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4938605, 0.4998064
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0253716, 1.0355630
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5145938, 0.5181644
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5261729, 0.5258648
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3923290, 0.3846605
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4244339, 0.4173191
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0449419, 1.0538874

Time for backsubstitution: 9.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 3, pos: 660

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3739477, upper bound: 0.3776179
time: 3.69 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3787069, upper bound: 0.3730398
time: 3.31 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8304319, 0.8289571
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6798899, 0.6898589
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7559953, 0.7480803
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4948823, 0.4987843
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0380154, 1.0231013
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5163856, 0.5160220
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5246599, 0.5273163
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3924105, 0.3844600
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4216516, 0.4201016
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0498266, 1.0483556

Time for backsubstitution: 9.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 660

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3739839, upper bound: 0.3773879
time: 3.74 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3787432, upper bound: 0.3728272
time: 3.31 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8289571, 0.8301392
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6884727, 0.6798899
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7480803, 0.7545710
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4987843, 0.4936479
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0231018, 1.0375953
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5163205, 0.5163854
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5219200, 0.5246599
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3844600, 0.3895641
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4201016, 0.4233629
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0479631, 1.0498261

Time for backsubstitution: 10.21 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.37 seconds

### Candidate
type: DSZ, layer: 3, pos: 660

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3728259, upper bound: 0.3787445
time: 4.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3773865, upper bound: 0.3739852
time: 4.36 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8303337, 0.8288064
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6880040, 0.6844809
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7478166, 0.7547998
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4998064, 0.4928547
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0355625, 1.0249848
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5184629, 0.5145938
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5211618, 0.5261726
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3846605, 0.3894654
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4173191, 0.4265845
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0536299, 1.0449414

Time for backsubstitution: 9.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 660

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3730385, upper bound: 0.3787083
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3776166, upper bound: 0.3739490
time: 3.45 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8290381, 0.8303509
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6880274, 0.6817214
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7496443, 0.7544317
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4998965, 0.4937702
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0240517, 1.0370641
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5141957, 0.5182118
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5236304, 0.5283458
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3864613, 0.3904091
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4242289, 0.4175242
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0441732, 1.0540085

Time for backsubstitution: 9.13 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 660

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3728259, upper bound: 0.3787445
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3773865, upper bound: 0.3739836
time: 3.89 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8304148, 0.8289928
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6872153, 0.6863124
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7493801, 0.7546954
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.5009186, 0.4927483
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0365143, 1.0244207
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.5163381, 0.5164202
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5221789, 0.5298588
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3866620, 0.3903276
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4214463, 0.4203067
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0497046, 1.0491242

Time for backsubstitution: 9.23 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 660
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 660

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3730385, upper bound: 0.3787083
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3776166, upper bound: 0.3739490
time: 3.50 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 16.67 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3739477, upper bound: 0.3776179
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3787069, upper bound: 0.3730398
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3739839, upper bound: 0.3773879
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3787432, upper bound: 0.3728272
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3739477, upper bound: 0.3776179
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3787069, upper bound: 0.3730398
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3739839, upper bound: 0.3773879
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3787432, upper bound: 0.3728272
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3728259, upper bound: 0.3787445
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3773865, upper bound: 0.3739852
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3730385, upper bound: 0.3787083
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3776166, upper bound: 0.3739490
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3728259, upper bound: 0.3787445
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3773865, upper bound: 0.3739836
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3730385, upper bound: 0.3787083
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 16.67
Output dim: 6, lower bound: -0.3776166, upper bound: 0.3739490

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8294444, 0.8260889
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6545875, 0.6525890
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7514358, 0.7434959
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4753444, 0.4810163
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0216198, 1.0346651
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4847567, 0.4887673
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5166162, 0.5173395
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3894193, 0.3825244
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4117062, 0.4157077
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0264702, 1.0259166

Time for backsubstitution: 9.07 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3739109, upper bound: 0.3768047
time: 3.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3731320, upper bound: 0.3775815
time: 3.82 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8249598, 0.8305736
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6506660, 0.6568778
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7502384, 0.7446928
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4740777, 0.4822803
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0231228, 1.0332932
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4892156, 0.4843764
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5196222, 0.5143340
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3894480, 0.3829071
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4128571, 0.4145570
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0249429, 1.0272026

Time for backsubstitution: 8.84 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3786701, upper bound: 0.3722284
time: 3.75 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3778904, upper bound: 0.3730032
time: 3.88 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8308024, 0.8247371
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6503403, 0.6534013
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7511716, 0.7437253
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4763663, 0.4802229
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0342636, 1.0222363
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4865487, 0.4866247
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5157975, 0.5187912
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3895009, 0.3823066
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4089236, 0.4189289
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0314903, 1.0203848

Time for backsubstitution: 9.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3739470, upper bound: 0.3765730
time: 3.56 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3731679, upper bound: 0.3773513
time: 3.30 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8263173, 0.8292217
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6464183, 0.6576899
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7499743, 0.7449217
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4750998, 0.4814869
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0357666, 1.0208645
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4910076, 0.4822339
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5188035, 0.5157855
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3895296, 0.3826894
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4100747, 0.4177783
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0299630, 1.0216713

Time for backsubstitution: 9.00 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3787063, upper bound: 0.3720157
time: 3.37 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3779263, upper bound: 0.3727906
time: 3.39 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8295250, 0.8263001
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6541436, 0.6544209
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7529984, 0.7433596
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4764569, 0.4811386
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0225706, 1.0341339
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4826322, 0.4905937
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5183276, 0.5210254
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3914208, 0.3833696
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4158336, 0.4098692
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0226808, 1.0300984

Time for backsubstitution: 9.18 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3739109, upper bound: 0.3768047
time: 3.65 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3731320, upper bound: 0.3775815
time: 3.81 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8250399, 0.8307853
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6502221, 0.6587095
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7518015, 0.7445560
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4751899, 0.4824026
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0240746, 1.0327621
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4870911, 0.4862028
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5213336, 0.5180197
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3914496, 0.3837523
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4169844, 0.4087185
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0211535, 1.0313849

Time for backsubstitution: 9.12 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3786701, upper bound: 0.3722288
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3778904, upper bound: 0.3730030
time: 3.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8308825, 0.8249235
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6495526, 0.6552330
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7527347, 0.7436233
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4774787, 0.4801165
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0352144, 1.0216722
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4844236, 0.4884512
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5168146, 0.5224769
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3915024, 0.3831689
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4130512, 0.4126518
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0275650, 1.0245671

Time for backsubstitution: 9.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3739470, upper bound: 0.3765730
time: 3.76 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3731679, upper bound: 0.3773513
time: 3.50 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8263979, 0.8294086
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6456306, 0.6595217
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7515373, 0.7448196
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4762120, 0.4813805
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0367184, 1.0202999
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4888825, 0.4840604
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5198206, 0.5194712
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3915311, 0.3835517
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4142021, 0.4115012
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0260377, 1.0258532

Time for backsubstitution: 9.34 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3787063, upper bound: 0.3720157
time: 3.70 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3779263, upper bound: 0.3727906
time: 3.43 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8294086, 0.8261061
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6581352, 0.6456307
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7448196, 0.7501135
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4813805, 0.4749774
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0202999, 1.0362983
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4843588, 0.4888827
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5140747, 0.5198205
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3835517, 0.3886846
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4115012, 0.4159132
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0254607, 1.0260382

Time for backsubstitution: 9.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3727893, upper bound: 0.3779261
time: 4.18 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3720142, upper bound: 0.3787077
time: 3.77 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8249235, 0.8305907
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6538465, 0.6495525
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7436233, 0.7513113
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4801166, 0.4762441
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0216713, 1.0347943
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4887495, 0.4844238
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5170807, 0.5168147
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3831689, 0.3886558
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4126518, 0.4147624
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0241747, 1.0275655

Time for backsubstitution: 9.04 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3773499, upper bound: 0.3731692
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3765715, upper bound: 0.3739484
time: 3.70 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8307853, 0.8247728
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6576664, 0.6502217
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7445560, 0.7503428
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4824026, 0.4741842
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0327625, 1.0236878
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4865012, 0.4870911
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5133165, 0.5213335
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3837523, 0.3885859
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4087186, 0.4191347
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0311279, 1.0211535

Time for backsubstitution: 9.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3730019, upper bound: 0.3778918
time: 3.43 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3722275, upper bound: 0.3786715
time: 3.52 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8263001, 0.8292575
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6533778, 0.6541435
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7433591, 0.7515402
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4811385, 0.4754509
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0341339, 1.0221839
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4908922, 0.4826322
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5163225, 0.5183275
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3833696, 0.3885572
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4098692, 0.4179839
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0298419, 1.0226808

Time for backsubstitution: 9.14 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3775801, upper bound: 0.3731334
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3768034, upper bound: 0.3739122
time: 3.56 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8294888, 0.8263173
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6576898, 0.6474625
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7463827, 0.7499743
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4824927, 0.4750997
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0212517, 1.0357671
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4822340, 0.4907091
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5157856, 0.5235064
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3855532, 0.3895296
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4156288, 0.4100746
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0216708, 1.0302200

Time for backsubstitution: 9.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3727893, upper bound: 0.3779261
time: 4.24 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3720142, upper bound: 0.3787077
time: 3.67 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8250041, 0.8308024
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6534011, 0.6513842
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7451859, 0.7511721
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4812288, 0.4763664
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0226231, 1.0342631
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4866247, 0.4862502
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5187911, 0.5205004
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3851705, 0.3895009
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4167794, 0.4089237
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0203848, 1.0317473

Time for backsubstitution: 9.09 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3773499, upper bound: 0.3731692
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3765715, upper bound: 0.3739484
time: 3.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8308654, 0.8249598
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6568778, 0.6520536
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7461185, 0.7502384
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4835150, 0.4740777
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0337133, 1.0231233
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4843764, 0.4889175
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5143336, 0.5250192
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3857539, 0.3894480
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4128460, 0.4128569
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0272026, 1.0253358

Time for backsubstitution: 9.17 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3730019, upper bound: 0.3778918
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3722275, upper bound: 0.3786715
time: 3.52 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8263807, 0.8294444
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6525891, 0.6559752
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7449222, 0.7514358
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4822509, 0.4753444
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0350857, 1.0216198
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4887671, 0.4844586
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5173396, 0.5220134
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3853711, 0.3894193
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4139966, 0.4117061
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0259166, 1.0268626

Time for backsubstitution: 9.08 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 613
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 613

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3775801, upper bound: 0.3731334
time: 3.59 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3768034, upper bound: 0.3739122
time: 3.78 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 16.69 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3739109, upper bound: 0.3768047
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3731320, upper bound: 0.3775815
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3786701, upper bound: 0.3722284
DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3778904, upper bound: 0.3730032
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3739470, upper bound: 0.3765730
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3731679, upper bound: 0.3773513
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3787063, upper bound: 0.3720157
DS_DSZ1_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3779263, upper bound: 0.3727906
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3739109, upper bound: 0.3768047
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3731320, upper bound: 0.3775815
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3786701, upper bound: 0.3722288
DS_DSZ1_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3778904, upper bound: 0.3730030
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3739470, upper bound: 0.3765730
DS_DSZ1_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3731679, upper bound: 0.3773513
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3787063, upper bound: 0.3720157
DS_DSZ1_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3779263, upper bound: 0.3727906
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3727893, upper bound: 0.3779261
DS_DSZ2_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3720142, upper bound: 0.3787077
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3773499, upper bound: 0.3731692
DS_DSZ2_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3765715, upper bound: 0.3739484
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3730019, upper bound: 0.3778918
DS_DSZ2_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3722275, upper bound: 0.3786715
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3775801, upper bound: 0.3731334
DS_DSZ2_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3768034, upper bound: 0.3739122
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3727893, upper bound: 0.3779261
DS_DSZ2_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3720142, upper bound: 0.3787077
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3773499, upper bound: 0.3731692
DS_DSZ2_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3765715, upper bound: 0.3739484
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3730019, upper bound: 0.3778918
DS_DSZ2_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3722275, upper bound: 0.3786715
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3775801, upper bound: 0.3731334
DS_DSZ2_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 5, time: 16.69
Output dim: 6, lower bound: -0.3768034, upper bound: 0.3739122

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8268132, 0.8235435
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6535704, 0.6514472
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7435067, 0.7363720
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4683578, 0.4742593
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0215755, 1.0346880
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4807098, 0.4838868
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5165844, 0.5172892
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3876716, 0.3797722
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4061685, 0.4119328
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0262666, 1.0256958

Time for backsubstitution: 9.11 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 3, pos: 1479

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3696914, upper bound: 0.3710552
time: 3.85 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3684962, upper bound: 0.3727333
time: 4.02 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8268991, 0.8234577
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6534460, 0.6515716
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7438676, 0.7355671
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4685876, 0.4740297
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0216422, 1.0346208
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4798763, 0.4846756
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5165658, 0.5173011
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3866673, 0.3805896
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4074662, 0.4101701
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0262494, 1.0256801

Time for backsubstitution: 9.10 seconds

### DS candidates at layer 3
type: DSZ, layer: 3, pos: 1479
type: DSZ, layer: 3, pos: 2901
type: DSZ, layer: 3, pos: 555
type: DSZ, layer: 3, pos: 2818
type: DSZ, layer: 3, pos: 1506
type: DSZ, layer: 3, pos: 75
type: DSZ, layer: 3, pos: 192
type: DSZ, layer: 3, pos: 1515
type: DSZ, layer: 3, pos: 3102
type: DSZ, layer: 3, pos: 2130
type: DSZ, layer: 3, pos: 38
type: DSZ, layer: 3, pos: 2858
type: DSZ, layer: 3, pos: 890
type: DSZ, layer: 3, pos: 1775
type: DSZ, layer: 3, pos: 1705
type: DSZ, layer: 3, pos: 780
type: DSZ, layer: 3, pos: 2396
type: DSZ, layer: 3, pos: 1406
type: DSZ, layer: 3, pos: 330
type: DSZ, layer: 3, pos: 76
type: DSZ, layer: 3, pos: 1228
type: DSZ, layer: 3, pos: 738
type: DSZ, layer: 3, pos: 2827
type: DSZ, layer: 3, pos: 1979
type: DSZ, layer: 3, pos: 151
type: DSZ, layer: 3, pos: 2356
type: DSZ, layer: 3, pos: 1847

Time for candidate selection: 0.22 seconds

### Candidate
type: DSZ, layer: 3, pos: 1479

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3689153, upper bound: 0.3718300
time: 3.36 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 6, lower bound: -0.3677219, upper bound: 0.3735079
time: 4.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -1.3027682, -0.1919377, -1.3027682, -0.1919377, -0.8223286, 0.8280282
1: -17.6880779, -16.2289352, -17.6880779, -16.2289352, -0.6496487, 0.6557360
2: -6.2674923, -4.9595528, -6.2674923, -4.9595528, -0.7423093, 0.7375689
3: -13.7172546, -12.5187931, -13.7172546, -12.5187931, -0.4670911, 0.4755232
4: -5.2706857, -4.0625572, -5.2706857, -4.0625572, -1.0230784, 1.0333161
5: -6.8115540, -5.8979759, -6.8115540, -5.8979759, -0.4851687, 0.4794959
6: 8.7676487, 9.8564320, 8.7676487, 9.8564320, -0.5195904, 0.5142832
7: -13.7728157, -12.5704174, -13.7728157, -12.5704174, -0.3877003, 0.3801550
8: -5.8852777, -4.9708009, -5.8852777, -4.9708009, -0.4073193, 0.4107822
9: -10.3968801, -8.9319839, -10.3968801, -8.9319839, -1.0247397, 1.0269818

Time for backsubstitution: 9.07 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 58.87 + 549.35 = 608.21 seconds
