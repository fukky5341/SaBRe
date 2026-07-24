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
execution time: IAR + RelationalAnalysis = 22.85 + 33.03 = 55.88 seconds
status: Status.UNKNOWN
relational distance
Output dim: 0, lower bound: -0.5411211, upper bound: 0.5411220

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6177
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 6177

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411192, upper bound: 0.5403046
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5403037, upper bound: 0.5411198
time: 3.72 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 7.24 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 7.24
Output dim: 0, lower bound: -0.5411192, upper bound: 0.5403046
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 7.24
Output dim: 0, lower bound: -0.5403037, upper bound: 0.5411198

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9406686, 0.9396138
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2499285, 1.2525015
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9732504, 0.9743729
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1950932, 1.1943769
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2330003, 1.2308855
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9743166, 0.9740274
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9605727, 0.9636374
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3981972, 1.3960843
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7550275, 0.7578721
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9503260, 0.9501023

Time for backsubstitution: 22.01 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4573

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411149, upper bound: 0.5391371
time: 3.57 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5399528, upper bound: 0.5402993
time: 6.47 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9396133, 0.9406686
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2525015, 1.2499285
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9743729, 0.9732506
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1943769, 1.1950932
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2308855, 1.2330003
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9740276, 0.9743164
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9636374, 0.9605727
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3960838, 1.3981967
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7578723, 0.7550275
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9501023, 0.9503260

Time for backsubstitution: 22.21 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4569

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5398107, upper bound: 0.5411186
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5403021, upper bound: 0.5406250
time: 3.54 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 29.20 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.20
Output dim: 0, lower bound: -0.5411149, upper bound: 0.5391371
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.20
Output dim: 0, lower bound: -0.5399528, upper bound: 0.5402993
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 29.20
Output dim: 0, lower bound: -0.5398107, upper bound: 0.5411186
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 29.20
Output dim: 0, lower bound: -0.5403021, upper bound: 0.5406250

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9452434, 0.9412589
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2340465, 1.2332172
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9633522, 0.9624927
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1877379, 1.1855597
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2338924, 1.2312226
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9484220, 0.9524508
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9493570, 0.9542847
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3916807, 1.3882294
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7537880, 0.7574995
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9338756, 0.9301350

Time for backsubstitution: 21.63 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4569

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5406198, upper bound: 0.5391357
time: 3.47 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411133, upper bound: 0.5386454
time: 3.26 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9423137, 0.9441886
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2306437, 1.2366195
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9613705, 0.9644747
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1862760, 1.1870222
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2333369, 1.2317781
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9527397, 0.9481330
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9512200, 0.9524217
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3903418, 1.3895688
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7546549, 0.7566328
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9303584, 0.9336522

Time for backsubstitution: 22.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5399520, upper bound: 0.5376183
time: 3.50 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5372706, upper bound: 0.5402995
time: 3.53 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9392815, 0.9405100
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2476163, 1.2475863
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9726758, 0.9696968
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1858230, 1.1909919
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2305527, 1.2323050
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9732141, 0.9739294
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9620166, 0.9571872
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3903713, 1.3954616
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7574825, 0.7542109
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9500322, 0.9501824

Time for backsubstitution: 22.07 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4573

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5398064, upper bound: 0.5399522
time: 3.45 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5386444, upper bound: 0.5411143
time: 3.43 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9394555, 0.9403365
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2501597, 1.2450428
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9708190, 0.9715533
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1902757, 1.1865392
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2301908, 1.2326674
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9736400, 0.9735031
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9602518, 0.9589520
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3933487, 1.3924842
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7570562, 0.7546377
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9499588, 0.9502554

Time for backsubstitution: 21.51 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5403013, upper bound: 0.5379200
time: 3.40 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5376199, upper bound: 0.5406242
time: 3.34 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 28.25 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.25
Output dim: 0, lower bound: -0.5406198, upper bound: 0.5391357
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.25
Output dim: 0, lower bound: -0.5411133, upper bound: 0.5386454
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.25
Output dim: 0, lower bound: -0.5399520, upper bound: 0.5376183
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.25
Output dim: 0, lower bound: -0.5372706, upper bound: 0.5402995
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.25
Output dim: 0, lower bound: -0.5398064, upper bound: 0.5399522
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.25
Output dim: 0, lower bound: -0.5386444, upper bound: 0.5411143
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 28.25
Output dim: 0, lower bound: -0.5403013, upper bound: 0.5379200
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 28.25
Output dim: 0, lower bound: -0.5376199, upper bound: 0.5406242

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

Time for backsubstitution: 21.50 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5406177, upper bound: 0.5346806
time: 5.35 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361519, upper bound: 0.5391325
time: 5.32 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

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

Time for backsubstitution: 22.15 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411125, upper bound: 0.5359629
time: 3.41 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5384249, upper bound: 0.5386446
time: 4.21 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9423137, 0.9441905
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2306271, 1.2366066
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9613762, 0.9644754
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1862712, 1.1870193
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2333279, 1.2317710
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9527392, 0.9481332
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9512200, 0.9524279
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3903422, 1.3895688
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7546549, 0.7566373
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9303589, 0.9336517

Time for backsubstitution: 21.60 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4569
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4569

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5394595, upper bound: 0.5376167
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5399504, upper bound: 0.5371187
time: 3.69 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9423156, 0.9441886
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2306309, 1.2366028
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9613709, 0.9644806
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1862726, 1.1870179
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2333302, 1.2317691
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9527402, 0.9481323
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9512262, 0.9524217
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3903422, 1.3895693
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7546592, 0.7566330
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9303579, 0.9336526

Time for backsubstitution: 21.61 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 4569

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5355176, upper bound: 0.5358428
time: 5.63 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5354939, upper bound: 0.5402964
time: 6.26 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

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

Time for backsubstitution: 22.41 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5398056, upper bound: 0.5372701
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5371179, upper bound: 0.5399514
time: 3.37 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

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

Time for backsubstitution: 23.37 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5386436, upper bound: 0.5384259
time: 3.51 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5359619, upper bound: 0.5411135
time: 3.70 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9394555, 0.9403384
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2501426, 1.2450299
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9708247, 0.9715531
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1902719, 1.1865368
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2301822, 1.2326612
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9736395, 0.9735041
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9602509, 0.9589572
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3933492, 1.3924842
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7570553, 0.7546418
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9499602, 0.9502554

Time for backsubstitution: 23.42 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4573
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 4573

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5402970, upper bound: 0.5367754
time: 3.54 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5391339, upper bound: 0.5379149
time: 5.31 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9394574, 0.9403365
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2501464, 1.2450261
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9708190, 0.9715586
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1902733, 1.1865354
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2301846, 1.2326589
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9736404, 0.9735031
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9602571, 0.9589510
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3933492, 1.3924842
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7570601, 0.7546375
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9499593, 0.9502568

Time for backsubstitution: 23.52 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550
type: DSZ, layer: 1, pos: 4573

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5358659, upper bound: 0.5361570
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5358423, upper bound: 0.5406221
time: 3.56 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.88 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5406177, upper bound: 0.5346806
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5361519, upper bound: 0.5391325
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5411125, upper bound: 0.5359629
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5384249, upper bound: 0.5386446
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5394595, upper bound: 0.5376167
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5399504, upper bound: 0.5371187
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5355176, upper bound: 0.5358428
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5354939, upper bound: 0.5402964
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5398056, upper bound: 0.5372701
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5371179, upper bound: 0.5399514
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5386436, upper bound: 0.5384259
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5359619, upper bound: 0.5411135
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5402970, upper bound: 0.5367754
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5391339, upper bound: 0.5379149
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5358659, upper bound: 0.5361570
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.88
Output dim: 0, lower bound: -0.5358423, upper bound: 0.5406221

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

Time for backsubstitution: 23.57 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5406169, upper bound: 0.5346782
time: 3.44 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361720, upper bound: 0.5346816
time: 3.62 seconds

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

Time for backsubstitution: 23.49 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 145

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 145

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361518, upper bound: 0.5347007
time: 5.55 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5361483, upper bound: 0.5391316
time: 6.56 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9450850, 0.9409287
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2316875, 1.2283187
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9598041, 0.9607954
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1836329, 1.1770034
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2331882, 1.2308826
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9480343, 0.9516377
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9459715, 0.9526701
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3889470, 1.3825173
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7529728, 0.7571149
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9337335, 0.9300644

Time for backsubstitution: 23.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 550

Time for candidate selection: 0.00 seconds

### Candidate
type: DSZ, layer: 1, pos: 550

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5411104, upper bound: 0.5341863
time: 4.66 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 0, lower bound: -0.5366547, upper bound: 0.5342100
time: 3.69 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: 9.2011337, 10.3698511, 9.2011337, 10.3698511, -0.9450870, 0.9409268
1: -16.9597416, -14.8090668, -16.9597416, -14.8090668, -1.2316914, 1.2283149
2: -6.5898213, -5.1094699, -6.5898213, -5.1094699, -0.9597983, 0.9608009
3: -8.5745993, -6.8540311, -8.5745993, -6.8540311, -1.1836343, 1.1770020
4: -10.3516960, -8.8085842, -10.3516960, -8.8085842, -1.2331905, 1.2308803
5: -2.0151374, -0.7350942, -2.0151374, -0.7350942, -0.9480348, 0.9516368
6: -1.3565187, -0.2398213, -1.3565187, -0.2398213, -0.9459777, 0.9526641
7: -8.2114935, -6.5324726, -8.2114935, -6.5324726, -1.3889470, 1.3825178
8: -1.7078848, -0.6203923, -1.7078848, -0.6203923, -0.7529771, 0.7571106
9: -4.0698876, -2.7048616, -4.0698876, -2.7048616, -0.9337320, 0.9300659

Time for backsubstitution: 23.71 seconds

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 55.88 + 552.67 = 608.54 seconds
