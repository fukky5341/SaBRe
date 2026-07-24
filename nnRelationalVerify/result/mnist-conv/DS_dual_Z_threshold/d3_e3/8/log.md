## Execution arguments:
Dataset: Dataset.MNIST
Network: ds/onnx/mnist_conv_exp.onnx
Relational property: GLOBAL_ROBUSTNESS
LP Analysis: True
Epsilon: 0.03515625
Delta epsilon: 0.01171875
execution index: (3, 3, 8)
Time budget: 600 seconds
Split limit: 100
Threshold: 0.716269655


## IAR start

### BASE IAR bounds
Layer (inp1_lb, inp1_ub, inp2_lb, inp2_ub, d_lb, d_ub)
0: (-15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.6121473, 1.6121478)
1: (-6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7759919, 1.7759929)
2: (-8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6092749, 1.6092749)
3: (-4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4462447, 1.4462445)
4: (-7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2081947, 1.2081950)
5: (-5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3884630, 1.3884630)
6: (-13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5876408, 1.5876408)
7: (2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2234151, 1.2234151)
8: (-0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.3073392, 1.3073397)
9: (-8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4662395, 1.4662395)

## BASE Relational Analysis

## BASE Result
execution time: IAR + RelationalAnalysis = 24.06 + 35.52 = 59.57 seconds
status: Status.UNKNOWN
relational distance
Output dim: 7, lower bound: -0.7198690, upper bound: 0.7198687

# Delta Split (DS) starts

## BFS DS instance: DS

Time for backsubstitution: 0.00 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6192
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.25 seconds

### Candidate
type: DSZ, layer: 1, pos: 6192

### Relational analysis ABCD of DS_DSZ1

#### Relational analysis ABCD result of DS_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198657, upper bound: 0.7142753
time: 3.58 seconds

### Relational analysis ABCD of DS_DSZ2

#### Relational analysis ABCD result of DS_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7142740, upper bound: 0.7198651
time: 6.32 seconds

## Summary of splitting (split count: 0)
- Time for DS candidates: 10.17 seconds
DS_DSZ1, status: Status.UNKNOWN, split count: 1, time: 10.17
Output dim: 7, lower bound: -0.7198657, upper bound: 0.7142753
DS_DSZ2, status: Status.UNKNOWN, split count: 1, time: 10.17
Output dim: 7, lower bound: -0.7142740, upper bound: 0.7198651

## BFS DS instance: DS_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5724177, 1.5790277
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7922955, 1.7954159
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6103306, 1.6105328
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4584475, 1.4564879
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2136455, 1.2146814
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3954854, 1.3943579
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5712547, 1.5676432
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2108376, 1.2083282
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2906146, 1.2933972
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4429021, 1.4382401

Time for backsubstitution: 22.82 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.35 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7184448, upper bound: 0.7142668
time: 4.92 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198595, upper bound: 0.7128290
time: 5.11 seconds

## BFS DS instance: DS_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5790277, 1.5724180
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7954159, 1.7922959
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6105328, 1.6103306
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4564877, 1.4584470
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2146811, 1.2136459
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3943582, 1.3954854
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5676432, 1.5712547
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2083285, 1.2108374
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2933974, 1.2906146
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4382401, 1.4429023

Time for backsubstitution: 22.65 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 4612
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.32 seconds

### Candidate
type: DSZ, layer: 1, pos: 4612

### Relational analysis ABCD of DS_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7128299, upper bound: 0.7198590
time: 3.94 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7142677, upper bound: 0.7184459
time: 4.11 seconds

## Summary of splitting (split count: 1)
- Time for DS candidates: 31.03 seconds
DS_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.03
Output dim: 7, lower bound: -0.7184448, upper bound: 0.7142668
DS_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.03
Output dim: 7, lower bound: -0.7198595, upper bound: 0.7128290
DS_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 2, time: 31.03
Output dim: 7, lower bound: -0.7128299, upper bound: 0.7198590
DS_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 2, time: 31.03
Output dim: 7, lower bound: -0.7142677, upper bound: 0.7184459

## BFS DS instance: DS_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5685515, 1.5728807
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7870693, 1.7921305
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6069417, 1.6051269
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4576149, 1.4551601
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2091248, 1.2074554
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3953891, 1.3942039
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5640779, 1.5631447
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2098262, 1.2076926
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2894878, 1.2915993
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4310937, 1.4308257

Time for backsubstitution: 23.22 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.38 seconds

### Candidate
type: DSZ, layer: 1, pos: 6140

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7184425, upper bound: 0.7126228
time: 4.72 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7168138, upper bound: 0.7142649
time: 9.48 seconds

## BFS DS instance: DS_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5662713, 1.5751607
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7890110, 1.7901893
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6049247, 1.6071439
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4571195, 1.4556558
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2064202, 1.2101605
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3953314, 1.3942611
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5667562, 1.5604665
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2102020, 1.2073169
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2888169, 1.2922704
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4354882, 1.4264314

Time for backsubstitution: 22.26 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 6140

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198572, upper bound: 0.7111848
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7182462, upper bound: 0.7128272
time: 3.78 seconds

## BFS DS instance: DS_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5751605, 1.5662713
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7901888, 1.7890110
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6071439, 1.6049247
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4556561, 1.4571195
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2101605, 1.2064199
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3942614, 1.3953311
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5604668, 1.5667562
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2073171, 1.2102022
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2922702, 1.2888169
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4264312, 1.4354885

Time for backsubstitution: 22.40 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 6140

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7128277, upper bound: 0.7182455
time: 5.93 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7111829, upper bound: 0.7198565
time: 4.29 seconds

## BFS DS instance: DS_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5728807, 1.5685513
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7921305, 1.7870688
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6051269, 1.6069417
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4551601, 1.4576149
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2074559, 1.2091250
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3942041, 1.3953886
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5631452, 1.5640779
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2076929, 1.2098265
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2915993, 1.2894878
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4308257, 1.4310937

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 6140
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 6140

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7142655, upper bound: 0.7168133
time: 3.95 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7126236, upper bound: 0.7184416
time: 4.51 seconds

## Summary of splitting (split count: 2)
- Time for DS candidates: 31.00 seconds
DS_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.00
Output dim: 7, lower bound: -0.7184425, upper bound: 0.7126228
DS_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.00
Output dim: 7, lower bound: -0.7168138, upper bound: 0.7142649
DS_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.00
Output dim: 7, lower bound: -0.7198572, upper bound: 0.7111848
DS_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.00
Output dim: 7, lower bound: -0.7182462, upper bound: 0.7128272
DS_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.00
Output dim: 7, lower bound: -0.7128277, upper bound: 0.7182455
DS_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.00
Output dim: 7, lower bound: -0.7111829, upper bound: 0.7198565
DS_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 3, time: 31.00
Output dim: 7, lower bound: -0.7142655, upper bound: 0.7168133
DS_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 3, time: 31.00
Output dim: 7, lower bound: -0.7126236, upper bound: 0.7184416

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5682173, 1.5720394
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7869272, 1.7917733
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6039734, 1.6039658
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4521289, 1.4530015
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2048588, 1.1966293
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3939986, 1.3936565
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5597796, 1.5614595
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2097628, 1.2075305
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2869339, 1.2905962
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4244599, 1.4282184

Time for backsubstitution: 22.27 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.33 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7184428, upper bound: 0.7102164
time: 4.51 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7159950, upper bound: 0.7126230
time: 4.48 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5677099, 1.5725465
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7867117, 1.7919884
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6057806, 1.6021590
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4554563, 1.4496741
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.1982989, 1.2031894
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3948417, 1.3928137
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5623927, 1.5588467
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2096641, 1.2076290
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2884846, 1.2890451
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4284863, 1.4241920

Time for backsubstitution: 22.36 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7168141, upper bound: 0.7118406
time: 3.78 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7143835, upper bound: 0.7142676
time: 3.62 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5659370, 1.5743194
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7888689, 1.7898321
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6019568, 1.6059828
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4516335, 1.4534969
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2021542, 1.1993344
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3939414, 1.3937140
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5624580, 1.5587814
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2101386, 1.2071548
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2862625, 1.2912672
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4288545, 1.4238238

Time for backsubstitution: 22.09 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198576, upper bound: 0.7087720
time: 3.48 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7174228, upper bound: 0.7111829
time: 3.72 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5654297, 1.5748265
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7886534, 1.7900476
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6037636, 1.6041756
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4549608, 1.4501698
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.1955938, 1.2058945
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3947840, 1.3928714
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5650711, 1.5561686
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2100399, 1.2072532
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2878141, 1.2897162
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4328809, 1.4197974

Time for backsubstitution: 22.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.30 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7182465, upper bound: 0.7103945
time: 3.60 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ2_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7158192, upper bound: 0.7128272
time: 3.71 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5748267, 1.5654297
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7900476, 1.7886539
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6041756, 1.6037636
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4501700, 1.4549608
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2058945, 1.1955938
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3928714, 1.3947840
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5561686, 1.5650711
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2072532, 1.2100401
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2897167, 1.2878137
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4197974, 1.4328809

Time for backsubstitution: 22.45 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7128280, upper bound: 0.7158190
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7103947, upper bound: 0.7182462
time: 3.73 seconds

## BFS DS instance: DS_DSZ2_DSZ1_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5743194, 1.5659370
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7898321, 1.7888689
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6059828, 1.6019568
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4534969, 1.4516335
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.1993341, 1.2021539
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3937140, 1.3939412
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5587811, 1.5624580
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2071550, 1.2101383
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2912674, 1.2862628
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4238238, 1.4288545

Time for backsubstitution: 22.66 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.31 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7111832, upper bound: 0.7174220
time: 4.34 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ1_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ1_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7087724, upper bound: 0.7198572
time: 5.41 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5725465, 1.5677099
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7919884, 1.7867117
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6021590, 1.6057806
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4496741, 1.4554563
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2031894, 1.1982989
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3928137, 1.3948414
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5588470, 1.5623927
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2076290, 1.2096643
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2890453, 1.2884848
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4241920, 1.4284863

Time for backsubstitution: 22.54 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.29 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7142658, upper bound: 0.7143834
time: 3.72 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7118407, upper bound: 0.7168136
time: 3.93 seconds

## BFS DS instance: DS_DSZ2_DSZ2_DSZ2

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5720396, 1.5682170
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7917728, 1.7869272
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.6039658, 1.6039734
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4530015, 1.4521289
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.1966295, 1.2048590
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3936563, 1.3939986
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5614595, 1.5597799
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.2075307, 1.2097626
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2905960, 1.2869337
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4282184, 1.4244599

Time for backsubstitution: 22.78 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 451
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 451

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ1

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ1
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7126239, upper bound: 0.7159963
time: 3.79 seconds

### Relational analysis ABCD of DS_DSZ2_DSZ2_DSZ2_DSZ2

#### Relational analysis ABCD result of DS_DSZ2_DSZ2_DSZ2_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7102172, upper bound: 0.7184428
time: 3.78 seconds

## Summary of splitting (split count: 3)
- Time for DS candidates: 30.61 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7184428, upper bound: 0.7102164
DS_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7159950, upper bound: 0.7126230
DS_DSZ1_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7168141, upper bound: 0.7118406
DS_DSZ1_DSZ1_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7143835, upper bound: 0.7142676
DS_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7198576, upper bound: 0.7087720
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7174228, upper bound: 0.7111829
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7182465, upper bound: 0.7103945
DS_DSZ1_DSZ2_DSZ2_DSZ2, status: Status.VERIFIED, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7158192, upper bound: 0.7128272
DS_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7128280, upper bound: 0.7158190
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7103947, upper bound: 0.7182462
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7111832, upper bound: 0.7174220
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7087724, upper bound: 0.7198572
DS_DSZ2_DSZ2_DSZ1_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7142658, upper bound: 0.7143834
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7118407, upper bound: 0.7168136
DS_DSZ2_DSZ2_DSZ2_DSZ1, status: Status.VERIFIED, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7126239, upper bound: 0.7159963
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.61
Output dim: 7, lower bound: -0.7102172, upper bound: 0.7184428

## BFS DS instance: DS_DSZ1_DSZ1_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5614710, 1.5667787
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7730370, 1.7801971
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.5933681, 1.5912404
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4459553, 1.4455941
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2039599, 1.1955483
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3888006, 1.3897610
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5392056, 1.5443168
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.1989050, 1.1941001
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2853804, 1.2887332
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4303789, 1.4307015

Time for backsubstitution: 22.38 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.27 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7184428, upper bound: 0.7102157
time: 4.12 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7160196, upper bound: 0.7102162
time: 4.42 seconds

## BFS DS instance: DS_DSZ1_DSZ1_DSZ2_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5609636, 1.5672858
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7728214, 1.7804127
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.5951748, 1.5894337
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4492826, 1.4422669
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.1973996, 1.2021081
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3896437, 1.3889182
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5418181, 1.5417037
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.1988068, 1.1941984
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2869310, 1.2871821
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4344053, 1.4266751

Time for backsubstitution: 22.44 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.28 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7168141, upper bound: 0.7118397
time: 4.01 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2
Status: Status.VERIFIED
Output dim: 7, lower bound: -0.7144082, upper bound: 0.7118425
time: 3.98 seconds

## BFS DS instance: DS_DSZ1_DSZ2_DSZ1_DSZ1

### Backsubstitution after applying DS history:
0: -15.3441200, -12.2636013, -15.3441200, -12.2636013, -1.5591908, 1.5690587
1: -6.8066797, -4.8410749, -6.8066797, -4.8410749, -1.7749786, 1.7782564
2: -8.3706236, -6.5761118, -8.3706236, -6.5761118, -1.5913510, 1.5932570
3: -4.6016669, -2.8533633, -4.6016669, -2.8533633, -1.4454603, 1.4460897
4: -7.5307088, -5.6608067, -7.5307088, -5.6608067, -1.2012544, 1.1982534
5: -5.9167237, -4.1329203, -5.9167237, -4.1329203, -1.3887434, 1.3898184
6: -13.9713326, -11.5294180, -13.9713326, -11.5294180, -1.5418839, 1.5416384
7: 2.7536235, 4.5407939, 2.7536235, 4.5407939, -1.1992807, 1.1937244
8: -0.9690433, 0.6157956, -0.9690433, 0.6157956, -1.2847090, 1.2894042
9: -8.3658676, -6.1971173, -8.3658676, -6.1971173, -1.4347734, 1.4263072

Time for backsubstitution: 22.33 seconds

### DS candidates at layer 1
type: DSZ, layer: 1, pos: 46

Time for candidate selection: 0.26 seconds

### Candidate
type: DSZ, layer: 1, pos: 46

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7198576, upper bound: 0.7087710
time: 3.82 seconds

### Relational analysis ABCD of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2

#### Relational analysis ABCD result of DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2
Status: Status.UNKNOWN
Output dim: 7, lower bound: -0.7174474, upper bound: 0.7087743
time: 3.88 seconds

## Summary of splitting (split count: 4)
- Time for DS candidates: 30.30 seconds
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.30
Output dim: 7, lower bound: -0.7184428, upper bound: 0.7102157
DS_DSZ1_DSZ1_DSZ1_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.30
Output dim: 7, lower bound: -0.7160196, upper bound: 0.7102162
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.30
Output dim: 7, lower bound: -0.7168141, upper bound: 0.7118397
DS_DSZ1_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.VERIFIED, split count: 5, time: 30.30
Output dim: 7, lower bound: -0.7144082, upper bound: 0.7118425
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ1, status: Status.UNKNOWN, split count: 5, time: 30.30
Output dim: 7, lower bound: -0.7198576, upper bound: 0.7087710
DS_DSZ1_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 5, time: 30.30
Output dim: 7, lower bound: -0.7174474, upper bound: 0.7087743
DS_DSZ1_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.30
Output dim: 7, lower bound: -0.7174228, upper bound: 0.7111829
DS_DSZ1_DSZ2_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.30
Output dim: 7, lower bound: -0.7182465, upper bound: 0.7103945
DS_DSZ2_DSZ1_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.30
Output dim: 7, lower bound: -0.7103947, upper bound: 0.7182462
DS_DSZ2_DSZ1_DSZ2_DSZ1, status: Status.UNKNOWN, split count: 4, time: 30.30
Output dim: 7, lower bound: -0.7111832, upper bound: 0.7174220
DS_DSZ2_DSZ1_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.30
Output dim: 7, lower bound: -0.7087724, upper bound: 0.7198572
DS_DSZ2_DSZ2_DSZ1_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.30
Output dim: 7, lower bound: -0.7118407, upper bound: 0.7168136
DS_DSZ2_DSZ2_DSZ2_DSZ2, status: Status.UNKNOWN, split count: 4, time: 30.30
Output dim: 7, lower bound: -0.7102172, upper bound: 0.7184428

## DS Result
status: Status.UNKNOWN
execution time: (base) + (ds) = 59.57 + 543.55 = 603.12 seconds
