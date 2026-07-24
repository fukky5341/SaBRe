## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.0234375
Delta epsilon: 0.0078125
execution index: (2, 3, 5)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.5357107799999999


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9421344, 0.9421344)
1: (-16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2560759, 1.2560759)
2: (-6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9759345, 0.9759345)
3: (-8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1960912, 1.1960917)
4: (-10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2359519, 1.2359519)
5: (-2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9747214, 0.9747217)
6: (-1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9679022, 0.9679022)
7: (-8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.4011202, 1.4011202)
8: (-1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7618291, 0.7618291)
9: (-4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9506359, 0.9506359)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 22.74 + 34.14 = 56.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5411211, upper bound: 0.5411220

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6177
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 4569

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5406260, upper bound: 0.5411192
time: 6.01 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411195, upper bound: 0.5406269
time: 3.65 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 9.88 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 9.88
Output dim: 0, lower bound: -0.5406260, upper bound: 0.5411192
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 9.88
Output dim: 0, lower bound: -0.5411195, upper bound: 0.5406269

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9418020, 0.9419763
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2511902, 1.2537336
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9742374, 0.9723809
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1875377, 1.1919904
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2356191, 1.2352571
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9739079, 0.9743342
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9662814, 0.9645166
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3954077, 1.3983850
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7614393, 0.7610128
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9505668, 0.9504933

Time for backsubstitution: 20.93 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6177
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 4573

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5406217, upper bound: 0.5399541
time: 3.83 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394622, upper bound: 0.5411151
time: 7.78 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9419761, 0.9418023
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2537336, 1.2511902
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9723806, 0.9742372
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1919904, 1.1875377
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2352571, 1.2356191
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9743342, 0.9739084
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9645166, 0.9662814
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3983850, 1.3954077
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7610130, 0.7614396
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9504938, 0.9505668

Time for backsubstitution: 22.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 6177
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 4573

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411153, upper bound: 0.5394632
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5399531, upper bound: 0.5406227
time: 3.64 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.53 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.53
Output dim: 0, lower bound: -0.5406217, upper bound: 0.5399541
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.53
Output dim: 0, lower bound: -0.5394622, upper bound: 0.5411151
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.53
Output dim: 0, lower bound: -0.5411153, upper bound: 0.5394632
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.53
Output dim: 0, lower bound: -0.5399531, upper bound: 0.5406227

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9463773, 0.9436216
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2353082, 1.2344494
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9643393, 0.9605010
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1801820, 1.1831722
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2365108, 1.2355933
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9480124, 0.9527564
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9550662, 0.9551644
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3888922, 1.3905306
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7602007, 0.7606406
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9341159, 0.9305255

Time for backsubstitution: 22.28 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6177
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6177

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5406198, upper bound: 0.5391357
time: 3.64 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5398064, upper bound: 0.5399522
time: 3.87 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9434476, 0.9465513
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2319059, 1.2378516
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9623575, 0.9624829
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1787195, 1.1846347
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2359557, 1.2361488
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9523301, 0.9484386
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9569292, 0.9533014
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3875532, 1.3918700
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7610672, 0.7597742
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9305987, 0.9340427

Time for backsubstitution: 22.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6177
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6177

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394603, upper bound: 0.5402988
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5386444, upper bound: 0.5411143
time: 3.62 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9465513, 0.9434476
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2378516, 1.2319059
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9624829, 0.9623575
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1846347, 1.1787195
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2361488, 1.2359552
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9484382, 0.9523301
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9533014, 0.9569292
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3918695, 1.3875532
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7597744, 0.7610674
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9340425, 0.9305985

Time for backsubstitution: 22.10 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6177
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 6177

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411133, upper bound: 0.5386454
time: 3.62 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5402978, upper bound: 0.5394612
time: 3.99 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9436216, 0.9463773
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2344494, 1.2353082
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9605012, 0.9643393
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1831722, 1.1801820
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2355933, 1.2365108
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9527564, 0.9480124
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9551644, 0.9550662
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3905306, 1.3888927
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7606409, 0.7602007
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9305253, 0.9341156

Time for backsubstitution: 22.14 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6177
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.20 seconds

### Candidate
type: DSZ, layer: 1, pos: 6177

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5399512, upper bound: 0.5398073
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5391347, upper bound: 0.5406208
time: 3.62 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 29.76 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 0, lower bound: -0.5406198, upper bound: 0.5391357
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 0, lower bound: -0.5398064, upper bound: 0.5399522
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 0, lower bound: -0.5394603, upper bound: 0.5402988
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 0, lower bound: -0.5386444, upper bound: 0.5411143
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 0, lower bound: -0.5411133, upper bound: 0.5386454
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 0, lower bound: -0.5402978, upper bound: 0.5394612
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 0, lower bound: -0.5399512, upper bound: 0.5398073
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 29.76
Output dim: 0, lower bound: -0.5391347, upper bound: 0.5406208

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9449115, 0.9411006
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2291617, 1.2308760
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9616551, 0.9589391
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1791840, 1.1814585
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2335596, 1.2305269
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9476080, 0.9520626
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9477363, 0.9508991
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3859692, 1.3854947
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7533991, 0.7566838
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9338045, 0.9299910

Time for backsubstitution: 21.47 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5406177, upper bound: 0.5346806
time: 5.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361519, upper bound: 0.5391325
time: 5.22 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9438562, 0.9421554
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2317348, 1.2283030
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9627776, 0.9578168
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1784682, 1.1821747
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2314448, 1.2326417
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9473186, 0.9523516
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9508009, 0.9478345
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3838568, 1.3876076
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7562439, 0.7538393
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9335814, 0.9302142

Time for backsubstitution: 22.25 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5398043, upper bound: 0.5354968
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5353477, upper bound: 0.5399490
time: 6.66 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9419818, 0.9440303
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2257595, 1.2342787
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9596734, 0.9609210
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1777220, 1.1829209
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2330041, 1.2310824
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9519258, 0.9477448
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9495993, 0.9490361
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3846292, 1.3868341
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7542660, 0.7558174
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9302874, 0.9335082

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394582, upper bound: 0.5358423
time: 3.98 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5350036, upper bound: 0.5402967
time: 3.93 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9409270, 0.9450850
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2283330, 1.2317052
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9607959, 0.9597986
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1770058, 1.1836367
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2308893, 1.2331972
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9516363, 0.9480338
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9526639, 0.9459715
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3825178, 1.3889465
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7571104, 0.7529726
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9300642, 0.9337313

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5386423, upper bound: 0.5366558
time: 3.93 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5341889, upper bound: 0.5411122
time: 4.14 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9450850, 0.9409270
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2317052, 1.2283325
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9597988, 0.9607956
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1836367, 1.1770058
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2331972, 1.2308893
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9480338, 0.9516368
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9459715, 0.9526639
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3889465, 1.3825173
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7529728, 0.7571106
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9337316, 0.9300640

Time for backsubstitution: 22.31 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.24 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411112, upper bound: 0.5341898
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5366548, upper bound: 0.5386433
time: 3.93 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9440303, 0.9419818
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2342787, 1.2257595
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9609213, 0.9596732
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1829209, 1.1777220
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2310829, 1.2330036
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9477448, 0.9519258
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9490361, 0.9495993
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3868341, 1.3846297
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7558172, 0.7542658
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9335079, 0.9302876

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.21 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5402957, upper bound: 0.5350036
time: 4.14 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5358415, upper bound: 0.5394591
time: 4.01 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9421558, 0.9438567
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2283030, 1.2317352
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9578171, 0.9627774
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1821747, 1.1784682
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2326417, 1.2314448
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9523516, 0.9473186
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9478345, 0.9508011
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3876066, 1.3838568
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7538393, 0.7562439
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9302144, 0.9335811

Time for backsubstitution: 22.19 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5399491, upper bound: 0.5353486
time: 3.97 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354958, upper bound: 0.5398053
time: 3.75 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9411006, 0.9449115
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2308764, 1.2291617
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9589391, 0.9616551
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1814585, 1.1791844
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2305274, 1.2335591
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9520626, 0.9476080
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9508991, 0.9477365
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3854952, 1.3859692
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7566841, 0.7533994
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9299908, 0.9338048

Time for backsubstitution: 22.23 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5391326, upper bound: 0.5361528
time: 4.00 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5346807, upper bound: 0.5406187
time: 3.76 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.26 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5406177, upper bound: 0.5346806
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5361519, upper bound: 0.5391325
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5398043, upper bound: 0.5354968
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5353477, upper bound: 0.5399490
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5394582, upper bound: 0.5358423
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5350036, upper bound: 0.5402967
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5386423, upper bound: 0.5366558
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5341889, upper bound: 0.5411122
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5411112, upper bound: 0.5341898
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5366548, upper bound: 0.5386433
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5402957, upper bound: 0.5350036
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5358415, upper bound: 0.5394591
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5399491, upper bound: 0.5353486
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5354958, upper bound: 0.5398053
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5391326, upper bound: 0.5361528
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.26
Output dim: 0, lower bound: -0.5346807, upper bound: 0.5406187

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9246879, 0.9168339
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.1856422, 1.1962667
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9353614, 0.9370289
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1760216, 1.1788235
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.1722569, 1.1794910
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9398661, 0.9427705
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9406061, 0.9416909
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3212767, 1.3315821
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7571216, 0.7594523
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9085209, 0.9089231

Time for backsubstitution: 22.34 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5406169, upper bound: 0.5346782
time: 3.71 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361720, upper bound: 0.5346816
time: 3.89 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9206443, 0.9208775
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.1945519, 1.1873565
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9397449, 0.9326456
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1765490, 1.1782961
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.1825233, 1.1692243
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9383159, 0.9443207
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9385281, 0.9437690
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3320560, 1.3208027
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7561679, 0.7604063
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9127367, 0.9047074

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361518, upper bound: 0.5347007
time: 5.81 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361483, upper bound: 0.5391316
time: 6.61 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9236326, 0.9178886
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.1882153, 1.1936932
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9364839, 0.9359066
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1753058, 1.1795397
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.1701422, 1.1816053
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9395766, 0.9430599
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9436707, 0.9386261
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3191643, 1.3336945
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7599664, 0.7566078
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9082978, 0.9091463

Time for backsubstitution: 22.29 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.23 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5398035, upper bound: 0.5354933
time: 4.46 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 0, lower bound: -0.5353679, upper bound: 0.5354967
time: 4.45 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9195895, 0.9219322
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.1971254, 1.1847832
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9408669, 0.9315231
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1758332, 1.1790118
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.1804085, 1.1713390
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9380269, 0.9446096
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9415927, 0.9407043
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3299437, 1.3229151
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7590127, 0.7575614
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9125130, 0.9049306

Time for backsubstitution: 22.20 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 56.88 + 559.27 = 616.15 seconds
